version = '0.6'

import os
import io
import math
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

# Pre-flight size threshold for the Opus output. If the estimated Opus file
# would exceed this, the raw PCM is split into chunks before encoding, giving
# sample-accurate offsets with no seek imprecision.
# Override with MAX_UPLOAD_MB env var (in MB). Default: 24 MB.
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_MB', '24')) * 1024 * 1024

# PCM properties — must match the ffmpeg encoding parameters below.
PCM_SAMPLE_RATE   = 16000  # Hz
PCM_BYTES_PER_SAMPLE = 2   # s16le

# Opus output bitrate used in all encode calls. Used to estimate Opus output
# size from PCM duration so we can decide whether to split before encoding.
OPUS_BITRATE_BPS = 12_000  # 12 kbps

# Cached result of whether the configured provider supports response_format=srt.
# None  = not yet probed    True = supports srt    False = use verbose_json
_provider_supports_srt: bool | None = None
_provider_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format HH:MM:SS,mmm."""
    millis = int(round(seconds * 1000))
    hours, millis   = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis    = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def verbose_json_to_segments(response) -> list[dict]:
    """Extract segments from a verbose_json response as a list of dicts."""
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in response.segments
    ]


def segments_to_srt(segments: list[dict]) -> str:
    """Render a flat list of segment dicts to a sequentially-numbered SRT string."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n"
            f"{seconds_to_srt_timestamp(seg['start'])} --> {seconds_to_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(lines)


def verbose_json_to_srt(response) -> str:
    return segments_to_srt(verbose_json_to_segments(response))


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def is_format_rejection(exc: BadRequestError) -> bool:
    """True when the provider rejected response_format=srt specifically."""
    return "response_format" in str(exc).lower()


def is_too_large_error(exc: Exception) -> bool:
    """
    True when the provider rejected the upload as too large (HTTP 413).
    Checks both the structured status code and raw message text, since some
    providers surface 413s through the error body rather than the HTTP status.
    """
    if isinstance(exc, APIStatusError) and exc.status_code == 413:
        return True
    msg = str(exc).lower()
    return "request_too_large" in msg or "request entity too large" in msg


# ---------------------------------------------------------------------------
# PCM chunking  (sample-accurate, no ffmpeg seeking)
# ---------------------------------------------------------------------------

def estimate_opus_bytes(pcm_bytes: bytes) -> int:
    """Estimate the Opus output size for a PCM input at the configured bitrate."""
    total_samples = len(pcm_bytes) // PCM_BYTES_PER_SAMPLE
    duration_secs = total_samples / PCM_SAMPLE_RATE
    return int(duration_secs * OPUS_BITRATE_BPS / 8)


def split_pcm_chunks(pcm_bytes: bytes, force: bool = False) -> list[tuple[bytes, float]]:
    """
    Split raw PCM bytes into chunks whose Opus output will fit within MAX_UPLOAD_BYTES.

    Returns a list of (pcm_chunk, start_offset_seconds) tuples.  Offsets are
    computed directly from the sample index and are therefore exact — there is
    no ffmpeg seeking, no frame-boundary rounding, and no accumulated error.

    force=True: split into at least 2 chunks even if the file is under the
    limit (used when the provider returned a 413 despite passing the pre-flight
    check, meaning its actual limit is lower than MAX_UPLOAD_BYTES).
    """
    total_samples   = len(pcm_bytes) // PCM_BYTES_PER_SAMPLE
    total_duration  = total_samples / PCM_SAMPLE_RATE

    if not force and estimate_opus_bytes(pcm_bytes) <= MAX_UPLOAD_BYTES:
        return [(pcm_bytes, 0.0)]

    # How many PCM samples can fit in one MAX_UPLOAD_BYTES Opus chunk?
    max_chunk_duration = (MAX_UPLOAD_BYTES * 8) / OPUS_BITRATE_BPS
    max_chunk_samples  = int(max_chunk_duration * PCM_SAMPLE_RATE)

    num_chunks     = max(2, math.ceil(total_samples / max_chunk_samples))
    samples_per_chunk = math.ceil(total_samples / num_chunks)
    bytes_per_chunk   = samples_per_chunk * PCM_BYTES_PER_SAMPLE

    print(
        f"{'Forced' if force else 'Pre-flight'} PCM split: "
        f"{len(pcm_bytes) / 1024 / 1024:.1f} MB PCM / {total_duration:.1f}s "
        f"→ {num_chunks} chunks of ~{total_duration / num_chunks:.1f}s each."
    )

    chunks = []
    for i in range(num_chunks):
        start_byte   = i * bytes_per_chunk
        end_byte     = min(start_byte + bytes_per_chunk, len(pcm_bytes))
        chunk_pcm    = pcm_bytes[start_byte:end_byte]
        # Offset is exact: sample index divided by sample rate.
        # No rounding, no seek imprecision, no accumulated error across chunks.
        start_offset = (i * samples_per_chunk) / PCM_SAMPLE_RATE
        chunk_opus_estimate = estimate_opus_bytes(chunk_pcm)
        print(
            f"  Chunk {i + 1}/{num_chunks}: "
            f"offset={start_offset:.3f}s  "
            f"pcm={len(chunk_pcm) / 1024 / 1024:.1f} MB  "
            f"opus≈{chunk_opus_estimate / 1024 / 1024:.2f} MB"
        )
        chunks.append((chunk_pcm, start_offset))

    return chunks


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------

def convert_pcm_to_opus(pcm_bytes: bytes) -> io.BytesIO:
    """
    Encode raw PCM bytes (s16le, 16kHz, mono) to Opus/OGG and return as BytesIO.
    Accepts bytes directly so PCM chunks can be encoded without wrapping.
    """
    try:
        out, _ = (
            ffmpeg.input("pipe:0", format="s16le", ar=PCM_SAMPLE_RATE, ac=1)
            .output(
                "pipe:1",
                format="opus",
                acodec="libopus",
                b=f"{OPUS_BITRATE_BPS // 1000}k",
                ac=1,
                ar=PCM_SAMPLE_RATE,
                application="voip",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, input=pcm_bytes)
        )
        opus_data = io.BytesIO(out)
        opus_data.name = "audio.ogg"
        opus_data.seek(0)
        return opus_data

    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e


# ---------------------------------------------------------------------------
# Core transcription — probes once, caches provider capability
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


def _transcribe_single(opus_data: io.BytesIO, task: str, language: str | None) -> str:
    """
    Transcribe a single Opus file and return an SRT string.
    Probes the provider on the first call to determine srt vs verbose_json support,
    then caches the result for all future calls.
    """
    global _provider_supports_srt

    # Fast path — provider capability already known
    if _provider_supports_srt is True:
        return _call_api(opus_data, task, language, "srt")

    if _provider_supports_srt is False:
        return verbose_json_to_srt(_call_api(opus_data, task, language, "verbose_json"))

    # First ever request: probe under lock
    with _provider_lock:
        # Re-check — another thread may have probed while we waited
        if _provider_supports_srt is True:
            return _call_api(opus_data, task, language, "srt")
        if _provider_supports_srt is False:
            return verbose_json_to_srt(_call_api(opus_data, task, language, "verbose_json"))

        try:
            response = _call_api(opus_data, task, language, "srt")
            _provider_supports_srt = True
            print("Provider supports response_format=srt — caching for future requests.")
            return response
        except BadRequestError as exc:
            if not is_format_rejection(exc):
                raise
            print(
                f"Provider rejected response_format=srt ({exc}). "
                "Using verbose_json — caching for future requests."
            )
            _provider_supports_srt = False
            return verbose_json_to_srt(_call_api(opus_data, task, language, "verbose_json"))


def _transcribe_chunk_verbose(opus_data: io.BytesIO, task: str, language: str | None) -> list[dict]:
    """
    Transcribe one chunk and return its segments as a list of dicts.
    Always uses verbose_json so callers get float timestamps they can offset.
    Sets _provider_supports_srt as a side-effect on the first ever call.
    """
    global _provider_supports_srt

    if _provider_supports_srt is None:
        with _provider_lock:
            if _provider_supports_srt is None:
                # Probe with srt to fill the cache, then fall through to verbose_json
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
                # Re-encode call happens below regardless of probe result
                opus_data.seek(0)

    response = _call_api(opus_data, task, language, "verbose_json")
    return verbose_json_to_segments(response)


def _transcribe_pcm_chunks(
    pcm_chunks: list[tuple[bytes, float]],
    task: str,
    language: str | None,
) -> str:
    """
    Encode each PCM chunk to Opus, transcribe it, apply the sample-accurate
    start offset to every segment, then merge all segments into one SRT.
    """
    all_segments: list[dict] = []

    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        opus_data = convert_pcm_to_opus(pcm_chunk)
        segments  = _transcribe_chunk_verbose(opus_data, task, language)
        for seg in segments:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segments)

    return segments_to_srt(all_segments)


def call_transcription(pcm_bytes: bytes, task: str, language: str | None) -> str:
    """
    Transcribe audio and always return an SRT string.

    Flow:
      1. Pre-flight PCM split check — if the estimated Opus output would exceed
         MAX_UPLOAD_BYTES, split the raw PCM now with sample-accurate offsets.
         Each PCM chunk is independently encoded to Opus and transcribed.
      2. Single-file attempt — convert the full PCM to Opus and call the API
         once using the cached format preference.
      3. 413 fallback — if the provider rejects the file, force-split the PCM
         and retry transparently.
    """
    # ── Step 1: pre-flight split ─────────────────────────────────────────────
    pcm_chunks = split_pcm_chunks(pcm_bytes)
    if len(pcm_chunks) > 1:
        return _transcribe_pcm_chunks(pcm_chunks, task, language)

    # ── Step 2: single-file attempt ──────────────────────────────────────────
    opus_data = convert_pcm_to_opus(pcm_bytes)
    try:
        return _transcribe_single(opus_data, task, language)

    # ── Step 3: 413 fallback ─────────────────────────────────────────────────
    except Exception as exc:
        if not is_too_large_error(exc):
            raise
        print(
            f"Provider returned 413 (file too large) — forcing PCM split and retrying. "
            f"Consider lowering MAX_UPLOAD_MB (currently {MAX_UPLOAD_BYTES // 1024 // 1024}) "
            f"to avoid this wasted round-trip in future."
        )
        forced_chunks = split_pcm_chunks(pcm_bytes, force=True)
        return _transcribe_pcm_chunks(forced_chunks, task, language)


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
    """
    Handle ASR requests from Bazarr. Reads the incoming raw PCM, splits it if
    needed, encodes each chunk to Opus, and returns a merged SRT to Bazarr.
    """
    try:
        start_time = time.time()

        # Read the raw PCM once. All chunking decisions are made on the PCM
        # bytes before any Opus encoding, giving sample-accurate offsets.
        pcm_bytes = audio_file.file.read()

        print(f"Got a {task} task from Bazarr ({len(pcm_bytes) / 1024 / 1024:.1f} MB PCM)")
        srt_content = call_transcription(pcm_bytes, task, language)

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
    print(
        f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} "
        f"using model: {whisper_model}, max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB (MAX_UPLOAD_MB)"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
