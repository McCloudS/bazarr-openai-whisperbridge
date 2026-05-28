version = '0.4'

import os
import io
import math
import subprocess
import json
import tempfile
import threading
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union
from openai import OpenAI, BadRequestError, APIStatusError
import ffmpeg
import time
import uvicorn

# Check if running inside a Docker container
in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

# Initialize FastAPI & OpenAI Client
app = FastAPI()
client = OpenAI()

force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', 'en')
whisper_model = os.getenv('WHISPER_MODEL', 'whisper-1')

# Pre-flight size check threshold. Set conservatively below 25 MB so that
# files we know are too large are split before the first API call, avoiding
# a wasted round-trip. Files that slip through (e.g. a provider with a lower
# undocumented limit) are caught by the 413 handler in call_transcription().
#
# Override with MAX_UPLOAD_MB env var (in megabytes) if your provider has a
# different limit. Examples:
#   MAX_UPLOAD_MB=25   # OpenAI / Groq default
#   MAX_UPLOAD_MB=10   # more conservative
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_MB', '24')) * 1024 * 1024

# Cached result of whether the configured provider supports response_format=srt.
# None  = not yet probed (first request will probe)
# True  = provider confirmed srt support
# False = provider rejected srt; use verbose_json + local conversion instead
_provider_supports_srt: bool | None = None
_provider_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert a float seconds value to an SRT timestamp string HH:MM:SS,mmm."""
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def verbose_json_to_segments(response) -> list[dict]:
    """
    Extract segments from a verbose_json response as a list of
    {'start': float, 'end': float, 'text': str} dicts.
    """
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in response.segments
    ]


def segments_to_srt(segments: list[dict]) -> str:
    """Render a flat list of segment dicts to a complete, sequentially-numbered SRT string."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = seconds_to_srt_timestamp(seg["start"])
        end_ts   = seconds_to_srt_timestamp(seg["end"])
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{seg['text']}\n")
    return "\n".join(lines)


def verbose_json_to_srt(response) -> str:
    """Convert a verbose_json response directly to an SRT string."""
    return segments_to_srt(verbose_json_to_segments(response))


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

def is_format_rejection(exc: BadRequestError) -> bool:
    """Return True when the error is specifically about an unsupported response_format."""
    return "response_format" in str(exc).lower()


def is_too_large_error(exc: Exception) -> bool:
    """
    Return True when the provider rejected the request because the file was too large (HTTP 413).

    Checks both the structured status code (openai.APIStatusError.status_code)
    and the raw message text, since some providers surface 413s through the
    error body rather than the HTTP status code itself.
    """
    if isinstance(exc, APIStatusError) and exc.status_code == 413:
        return True
    msg = str(exc).lower()
    return "request_too_large" in msg or "request entity too large" in msg


# ---------------------------------------------------------------------------
# Audio chunking helpers
# ---------------------------------------------------------------------------

def get_opus_duration(opus_data: io.BytesIO) -> float:
    """
    Return the duration of an Opus/OGG stream in seconds using ffprobe.
    Uses a named temp file since ffprobe duration detection is more reliable
    with seekable file input than with stdin pipes.
    """
    opus_data.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(opus_data.read())
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", tmp_path],
            capture_output=True,
            check=True,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    finally:
        os.unlink(tmp_path)
        opus_data.seek(0)


def extract_opus_chunk(opus_data: io.BytesIO, start_secs: float, duration_secs: float) -> io.BytesIO:
    """
    Extract a time slice from an Opus/OGG stream and return it as a new BytesIO.
    Re-encodes the slice to Opus so the output is a self-contained valid file.
    """
    opus_data.seek(0)
    out, _ = (
        ffmpeg
        .input("pipe:0", format="ogg", ss=start_secs, t=duration_secs)
        .output("pipe:1", format="opus", acodec="libopus", b="12k", ac=1, ar=16000, application="voip")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True, input=opus_data.read())
    )
    chunk = io.BytesIO(out)
    chunk.name = "chunk.ogg"
    chunk.seek(0)
    return chunk


def split_into_chunks(opus_data: io.BytesIO, force: bool = False) -> list[tuple[io.BytesIO, float]]:
    """
    Split opus_data into equal-duration chunks that each fit within MAX_UPLOAD_BYTES.

    force=False (default): only split if the file actually exceeds MAX_UPLOAD_BYTES.
                           Returns a single-element list when no split is needed.
    force=True:            split regardless of file size (used after a 413 to retry
                           a file that passed the pre-flight check but was still
                           rejected by the provider at request time).

    Returns a list of (chunk_bytes, start_offset_secs) tuples.
    """
    opus_size = opus_data.getbuffer().nbytes

    if not force and opus_size <= MAX_UPLOAD_BYTES:
        opus_data.seek(0)
        return [(opus_data, 0.0)]

    # Determine how many chunks are needed.
    # When force=True and the file is under MAX_UPLOAD_BYTES we still produce at
    # least 2 chunks, so the retry is meaningfully different from the attempt that
    # just 413'd (the provider's actual limit must be lower than MAX_UPLOAD_BYTES).
    num_chunks = max(2, math.ceil(opus_size / MAX_UPLOAD_BYTES))
    total_duration = get_opus_duration(opus_data)
    chunk_duration = total_duration / num_chunks

    print(
        f"{'Forced split' if force else 'Pre-flight split'}: "
        f"{opus_size / 1024 / 1024:.1f} MB / {total_duration:.1f}s → "
        f"{num_chunks} chunks of ~{chunk_duration:.1f}s each."
    )

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        # Give the last chunk a slightly generous duration so ffmpeg runs
        # to the natural end of the stream without clipping the final word.
        duration = chunk_duration if i < num_chunks - 1 else (total_duration - start + 1)
        chunk = extract_opus_chunk(opus_data, start_secs=start, duration_secs=duration)
        print(f"  Chunk {i + 1}/{num_chunks}: start={start:.1f}s  size={chunk.getbuffer().nbytes / 1024 / 1024:.2f} MB")
        chunks.append((chunk, start))

    return chunks


# ---------------------------------------------------------------------------
# Core transcription — probes once, caches provider capability, handles chunks
# ---------------------------------------------------------------------------

def _call_api(opus_data: io.BytesIO, task: str, language: str | None, fmt: str):
    """Raw API call with an explicit response_format."""
    opus_data.seek(0)
    if task == "transcribe":
        return client.audio.transcriptions.create(
            model=whisper_model,
            file=opus_data,
            response_format=fmt,
            language=language,
        )
    else:
        return client.audio.translations.create(
            model=whisper_model,
            file=opus_data,
            response_format=fmt,
        )


def _transcribe_single_verbose(opus_data: io.BytesIO, task: str, language: str | None) -> list[dict]:
    """
    Transcribe one chunk and return its segments as a list of dicts.
    Always uses verbose_json so callers get float timestamps they can offset.
    Sets _provider_supports_srt as a side-effect on the first call.
    """
    global _provider_supports_srt

    def _probe():
        global _provider_supports_srt
        # Re-check inside lock — another thread may have beaten us here.
        if _provider_supports_srt is not None:
            return
        try:
            _call_api(opus_data, task, language, "srt")
            _provider_supports_srt = True
            print("Provider supports response_format=srt — caching for future requests.")
        except BadRequestError as exc:
            if not is_format_rejection(exc):
                raise
            print(
                f"Provider rejected response_format=srt ({exc}). "
                "Using verbose_json — caching for future requests."
            )
            _provider_supports_srt = False

    if _provider_supports_srt is None:
        with _provider_lock:
            _probe()

    response = _call_api(opus_data, task, language, "verbose_json")
    return verbose_json_to_segments(response)


def _transcribe_chunks(chunks: list[tuple[io.BytesIO, float]], task: str, language: str | None) -> str:
    """
    Transcribe a pre-split list of chunks and merge into a single SRT.
    Always uses verbose_json so timestamps can be offset and merged correctly.
    """
    all_segments: list[dict] = []

    for idx, (chunk_data, start_offset) in enumerate(chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(chunks)} (offset={start_offset:.1f}s) ...")
        segments = _transcribe_single_verbose(chunk_data, task, language)
        for seg in segments:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segments)

    return segments_to_srt(all_segments)


def call_transcription(opus_data: io.BytesIO, task: str, language: str | None) -> str:
    """
    Transcribe audio and always return an SRT string.

    Flow:
      1. Pre-flight size check — if the file already exceeds MAX_UPLOAD_BYTES,
         skip straight to chunked transcription (avoids a guaranteed failed call).
      2. Single-file attempt — try the whole file in one API call using the
         cached format preference (srt or verbose_json).
      3. 413 fallback — if the provider rejects the file as too large, split it
         into chunks and retry transparently.  This catches providers whose actual
         limit is lower than MAX_UPLOAD_BYTES and files where the pre-flight check
         passed but the provider still rejected it.
    """
    global _provider_supports_srt

    # ── Step 1: pre-flight size check ──────────────────────────────────────
    chunks = split_into_chunks(opus_data)          # force=False: only splits if needed
    if len(chunks) > 1:
        return _transcribe_chunks(chunks, task, language)

    # ── Step 2: single-file attempt ─────────────────────────────────────────
    single_data, _ = chunks[0]

    try:
        # Fast path — provider capability already known
        if _provider_supports_srt is True:
            return _call_api(single_data, task, language, "srt")

        if _provider_supports_srt is False:
            return verbose_json_to_srt(_call_api(single_data, task, language, "verbose_json"))

        # First ever request: probe under lock
        with _provider_lock:
            if _provider_supports_srt is None:
                try:
                    response = _call_api(single_data, task, language, "srt")
                    _provider_supports_srt = True
                    print("Provider supports response_format=srt — caching for future requests.")
                    return response
                except BadRequestError as exc:
                    if not is_format_rejection(exc):
                        raise
                    print(
                        f"Provider rejected response_format=srt ({exc}). "
                        "Retrying with verbose_json — caching for future requests."
                    )
                    _provider_supports_srt = False

            # Another thread completed the probe while we waited; fall through
            if _provider_supports_srt is True:
                return _call_api(single_data, task, language, "srt")
            return verbose_json_to_srt(_call_api(single_data, task, language, "verbose_json"))

    # ── Step 3: 413 fallback ────────────────────────────────────────────────
    except Exception as exc:
        if not is_too_large_error(exc):
            raise

        print(
            f"Provider returned 413 (file too large) — splitting into chunks and retrying. "
            f"Consider lowering MAX_UPLOAD_MB (currently {MAX_UPLOAD_BYTES // 1024 // 1024}) "
            f"to avoid this wasted round-trip in future."
        )
        forced_chunks = split_into_chunks(opus_data, force=True)
        return _transcribe_chunks(forced_chunks, task, language)


# ---------------------------------------------------------------------------
# FFmpeg PCM → Opus conversion
# ---------------------------------------------------------------------------

def convert_pcm_to_opus_in_memory(input_data) -> io.BytesIO:
    """
    Converts a raw PCM bytestream to Opus format and returns it as an in-memory BytesIO object.
    """
    try:
        input_data.seek(0)
        out, _ = (
            ffmpeg.input("pipe:0", format="s16le", ar=16000, ac=1)
            .output("pipe:1", format="opus", ac=1, ar=16000, acodec="libopus", b="12k", application="voip")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, input=input_data.read())
        )
        opus_data = io.BytesIO(out)
        opus_data.name = "file.ogg"
        opus_data.seek(0)
        return opus_data

    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}") from e


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    return {"version": f"Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version}"}


@app.post("/detect-language")
def detect_language():
    print(f"Forced detected language to {force_detected_language_to}")
    return {
        "detected_language": f"Forced to {force_detected_language_to} from WhisperBridge",
        "language_code": force_detected_language_to,
    }

@app.post("/asr")
def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
):
    try:
        start_time = time.time()
        opus_data = convert_pcm_to_opus_in_memory(audio_file.file)

        print(f"Got a {task} task from Bazarr")
        srt_content = call_transcription(opus_data, task, language)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        print(
            f"Transcription of '{video_file}' complete in {minutes}m {seconds}s."
            if video_file
            else f"Transcription complete in {minutes}m {seconds}s."
        )

        if srt_content:
            return StreamingResponse(
                iter([srt_content]),
                media_type="text/plain",
                headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} using model: {whisper_model}, max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB (MAX_UPLOAD_MB)")
    uvicorn.run(app, host="0.0.0.0", port=9000)
