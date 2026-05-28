version = '0.99'

import os
import io
import math
import traceback
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union
from openai import OpenAI, APIStatusError
import ffmpeg
import time
import uvicorn

in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

app = FastAPI()
client = OpenAI()

force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', 'en')
whisper_model = os.getenv('WHISPER_MODEL', 'whisper-1')

MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_MB', '24')) * 1024 * 1024
OPUS_BITRATE_BPS = int(os.getenv('OPUS_BITRATE_KBPS', '24')) * 1000
OPUS_APPLICATION = "audio"

# Maximum characters per subtitle line. Lines longer than this are split at
# the nearest word boundary using the word-level timestamps from the API.
MAX_LINE_LENGTH  = int(os.getenv('MAX_LINE_LENGTH', '42'))

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis   = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis    = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def verbose_json_to_segments(response) -> list[dict]:
    """
    Extract segments from a verbose_json response.

    Handles two provider formats:
      - Groq:   words in a top-level response.words array
      - OpenAI: words nested inside each segment object

    When top-level words are present we ignore the provider's segment
    boundaries and rebuild segments ourselves using split_segments(), giving
    us precise word-boundary splits at the desired line length.
    """
    # --- Groq: top-level words array ---
    top_words = [
        {
            "word":  getattr(w, "word",  "").strip(),
            "start": getattr(w, "start", None),
            "end":   getattr(w, "end",   None),
        }
        for w in (getattr(response, "words", None) or [])
        if getattr(w, "start", None) is not None
        and getattr(w, "end",   None) is not None
    ]
    if top_words:
        top_words.sort(key=lambda w: w["start"])
        print(f"Using {len(top_words)} top-level words — splitting into segments.")
        return split_segments(top_words)

    # --- OpenAI: segment-nested words ---
    raw_segments = getattr(response, "segments", None) or []
    segments = []
    for seg in raw_segments:
        start = getattr(seg, "start", None)
        end   = getattr(seg, "end",   None)
        text  = (getattr(seg, "text", "") or "").strip()
        if start is None or end is None:
            continue
        words = [
            {
                "word":  getattr(w, "word", "").strip(),
                "start": w.start,
                "end":   w.end,
            }
            for w in (getattr(seg, "words", None) or [])
            if getattr(w, "start", None) is not None
            and getattr(w, "end",   None) is not None
        ]
        if words:
            segments.extend(split_segments(words))
        else:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def split_segments(words: list[dict]) -> list[dict]:
    """
    Group a flat list of word dicts into subtitle segments, splitting when
    the accumulated line would exceed MAX_LINE_LENGTH characters.

    Each segment's start/end is taken directly from the word timestamps so
    the subtitle appears exactly when the first word is spoken and disappears
    when the last word ends — no padding, no silence bleed.
    """
    segments = []
    current_words: list[dict] = []
    current_len = 0

    for word in words:
        w_text = word["word"]
        # +1 for the space between words (except the first word)
        added = len(w_text) + (1 if current_words else 0)

        if current_words and current_len + added > MAX_LINE_LENGTH:
            # Flush current segment
            segments.append(_words_to_segment(current_words))
            current_words = [word]
            current_len   = len(w_text)
        else:
            current_words.append(word)
            current_len += added

    if current_words:
        segments.append(_words_to_segment(current_words))

    return segments


def _words_to_segment(words: list[dict]) -> dict:
    return {
        "start": words[0]["start"],
        "end":   words[-1]["end"],
        "text":  " ".join(w["word"] for w in words),
    }


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n"
            f"{seconds_to_srt_timestamp(seg['start'])} --> {seconds_to_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def is_too_large_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError) and exc.status_code == 413:
        return True
    msg = str(exc).lower()
    return "request_too_large" in msg or "request entity too large" in msg


# ---------------------------------------------------------------------------
# Opus encoding
# ---------------------------------------------------------------------------

def encode_pcm_to_opus(pcm_bytes: bytes) -> io.BytesIO:
    try:
        out, err = (
            ffmpeg.input("pipe:0", format="s16le", ar=PCM_SAMPLE_RATE, ac=PCM_NUM_CHANNELS)
            .output(
                "pipe:1",
                format="opus",
                acodec="libopus",
                b=f"{OPUS_BITRATE_BPS // 1000}k",
                ac=PCM_NUM_CHANNELS,
                ar=PCM_SAMPLE_RATE,
                application=OPUS_APPLICATION,
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, input=pcm_bytes)
        )
        if not out:
            stderr_msg = err.decode(errors="replace") if err else "(no stderr)"
            raise RuntimeError(f"FFmpeg produced no output. stderr: {stderr_msg}")
        opus = io.BytesIO(out)
        opus.name = "audio.ogg"
        opus.seek(0)
        return opus
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode(errors="replace") if e.stderr else "(no stderr)"
        raise RuntimeError(f"FFmpeg error: {stderr_msg}") from e


# ---------------------------------------------------------------------------
# PCM chunking  (sample-accurate offsets)
# ---------------------------------------------------------------------------

def split_pcm_into_chunks(pcm_bytes: bytes, num_chunks: int) -> list[tuple[bytes, float]]:
    """Split raw PCM into num_chunks equal pieces with sample-accurate offsets."""
    total_samples     = len(pcm_bytes) // PCM_BYTES_PER_SAMPLE
    total_duration    = total_samples / PCM_SAMPLE_RATE
    samples_per_chunk = math.ceil(total_samples / num_chunks)
    bytes_per_chunk   = samples_per_chunk * PCM_BYTES_PER_SAMPLE

    print(
        f"Splitting {len(pcm_bytes) / 1024 / 1024:.1f} MB PCM / {total_duration:.1f}s "
        f"→ {num_chunks} chunks of ~{total_duration / num_chunks:.1f}s each."
    )

    chunks = []
    for i in range(num_chunks):
        start_byte   = i * bytes_per_chunk
        end_byte     = min(start_byte + bytes_per_chunk, len(pcm_bytes))
        chunk_pcm    = pcm_bytes[start_byte:end_byte]
        start_offset = (i * samples_per_chunk) / PCM_SAMPLE_RATE
        print(f"  Chunk {i + 1}/{num_chunks}: offset={start_offset:.3f}s  pcm={len(chunk_pcm) / 1024 / 1024:.1f} MB")
        chunks.append((chunk_pcm, start_offset))

    return chunks


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

def _call_api(opus: io.BytesIO, task: str, language: str | None) -> list[dict]:
    """Call the provider with verbose_json + word timestamps. Returns segments."""
    opus.seek(0)
    if task == "transcribe":
        response = client.audio.transcriptions.create(
            model=whisper_model,
            file=opus,
            response_format="verbose_json",
            language=language,
            timestamp_granularities=["word", "segment"],
        )
    else:
        response = client.audio.translations.create(
            model=whisper_model,
            file=opus,
            response_format="verbose_json",
        )
    return verbose_json_to_segments(response)


def _transcribe_pcm_chunks(
    pcm_chunks: list[tuple[bytes, float]],
    task: str,
    language: str | None,
) -> str:
    """Encode each PCM chunk, transcribe, apply sample-accurate offset, merge to SRT."""
    all_segments: list[dict] = []
    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        segs = _call_api(encode_pcm_to_opus(pcm_chunk), task, language)
        for seg in segs:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segs)
    return segments_to_srt(all_segments)


def call_transcription(pcm_bytes: bytes, task: str, language: str | None) -> str:
    opus      = encode_pcm_to_opus(pcm_bytes)
    opus_size = opus.getbuffer().nbytes
    print(
        f"Encoded to Opus: {opus_size / 1024 / 1024:.1f} MB "
        f"(from {len(pcm_bytes) / 1024 / 1024:.1f} MB PCM, "
        f"{len(pcm_bytes) / opus_size:.0f}:1 compression)"
    )

    if opus_size <= MAX_UPLOAD_BYTES:
        try:
            return segments_to_srt(_call_api(opus, task, language))
        except Exception as exc:
            if not is_too_large_error(exc):
                raise
            print(
                f"Provider returned 413 despite {opus_size / 1024 / 1024:.1f} MB Opus file. "
                f"Consider lowering MAX_UPLOAD_MB (currently {MAX_UPLOAD_BYTES // 1024 // 1024}). "
                f"Splitting into chunks and retrying."
            )
            num_chunks = max(2, math.ceil(opus_size / MAX_UPLOAD_BYTES) + 1)
            return _transcribe_pcm_chunks(
                split_pcm_into_chunks(pcm_bytes, num_chunks), task, language
            )

    num_chunks = max(2, math.ceil(opus_size / MAX_UPLOAD_BYTES))
    print(f"Opus file ({opus_size / 1024 / 1024:.1f} MB) exceeds limit — splitting into {num_chunks} chunks.")
    return _transcribe_pcm_chunks(
        split_pcm_into_chunks(pcm_bytes, num_chunks), task, language
    )


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
        pcm_bytes  = audio_file.file.read()

        print(f"Got a {task} task from Bazarr ({len(pcm_bytes) / 1024 / 1024:.1f} MB PCM)")
        srt_content = call_transcription(pcm_bytes, task, language)

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(
            f"Transcription of '{video_file}' complete in {mins}m {secs}s."
            if video_file else f"Transcription complete in {mins}m {secs}s."
        )

        return StreamingResponse(
            iter([srt_content or ""]),
            media_type="text/plain",
            headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(
        f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} | "
        f"model: {whisper_model} | "
        f"opus: {OPUS_BITRATE_BPS // 1000} kbps | "
        f"max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB | "
        f"max line: {MAX_LINE_LENGTH} chars"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
