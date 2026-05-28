version = '0.7'

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

in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

app = FastAPI()
client = OpenAI()

force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', 'en')
whisper_model = os.getenv('WHISPER_MODEL', 'whisper-1')

MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_MB', '24')) * 1024 * 1024

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

# 24 kbps is the sweet spot for Whisper: a 2-hour film encodes to ~20 MB
# (comfortably under the 25 MB provider limit), and quality is perceptually
# transparent for speech at this rate.
# application="audio" preserves the full audio signal without the noise
# suppression and VAD that application="voip" applies — important for quiet
# dialogue, accents, and anything Whisper needs to hear unmodified.
OPUS_BITRATE_BPS  = int(os.getenv('OPUS_BITRATE_KBPS', '24')) * 1000
OPUS_APPLICATION  = "audio"

_provider_supports_srt: bool | None = None
_provider_lock = threading.Lock()


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
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in response.segments
    ]


def segments_to_srt(segments: list[dict]) -> str:
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
    return "response_format" in str(exc).lower()


def is_too_large_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError) and exc.status_code == 413:
        return True
    msg = str(exc).lower()
    return "request_too_large" in msg or "request entity too large" in msg


# ---------------------------------------------------------------------------
# Opus encoding
# ---------------------------------------------------------------------------

def encode_pcm_to_opus(pcm_bytes: bytes) -> io.BytesIO:
    """
    Encode raw PCM bytes to Opus/OGG and return a named BytesIO.

    Settings chosen for Whisper transcription quality:
      - 24 kbps: perceptually transparent for speech; 2-hour film ≈ 20 MB
      - application=audio: preserves the full signal without the noise
        suppression / VAD that voip mode applies
      - 16 kHz mono: matches Whisper's internal sample rate exactly
    """
    try:
        out, _ = (
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
        opus = io.BytesIO(out)
        opus.name = "audio.ogg"
        opus.seek(0)
        return opus
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e


# ---------------------------------------------------------------------------
# PCM chunking  (sample-accurate offsets)
# ---------------------------------------------------------------------------

def split_pcm_into_chunks(pcm_bytes: bytes, num_chunks: int) -> list[tuple[bytes, float]]:
    """
    Split raw PCM bytes into exactly num_chunks equal pieces.

    Offsets are computed from sample index / sample rate — exact to the
    sample, with no encoding delay or seek imprecision across chunks.
    """
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

def _call_api(opus: io.BytesIO, task: str, language: str | None, fmt: str):
    opus.seek(0)
    if task == "transcribe":
        return client.audio.transcriptions.create(
            model=whisper_model, file=opus, response_format=fmt, language=language,
        )
    else:
        return client.audio.translations.create(
            model=whisper_model, file=opus, response_format=fmt,
        )


def _transcribe_single(opus: io.BytesIO, task: str, language: str | None) -> str:
    """Transcribe a single Opus file, probing and caching the provider's format support."""
    global _provider_supports_srt

    if _provider_supports_srt is True:
        return _call_api(opus, task, language, "srt")
    if _provider_supports_srt is False:
        return verbose_json_to_srt(_call_api(opus, task, language, "verbose_json"))

    with _provider_lock:
        if _provider_supports_srt is True:
            return _call_api(opus, task, language, "srt")
        if _provider_supports_srt is False:
            return verbose_json_to_srt(_call_api(opus, task, language, "verbose_json"))

        try:
            response = _call_api(opus, task, language, "srt")
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
            return verbose_json_to_srt(_call_api(opus, task, language, "verbose_json"))


def _transcribe_chunk_verbose(opus: io.BytesIO, task: str, language: str | None) -> list[dict]:
    """Transcribe one chunk with verbose_json. Sets _provider_supports_srt as a side-effect."""
    global _provider_supports_srt

    if _provider_supports_srt is None:
        with _provider_lock:
            if _provider_supports_srt is None:
                try:
                    _call_api(opus, task, language, "srt")
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

    return verbose_json_to_segments(_call_api(opus, task, language, "verbose_json"))


def _transcribe_pcm_chunks(
    pcm_chunks: list[tuple[bytes, float]],
    task: str,
    language: str | None,
) -> str:
    """Encode each PCM chunk to Opus, transcribe, apply offset, and merge into one SRT."""
    all_segments: list[dict] = []
    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        opus = encode_pcm_to_opus(pcm_chunk)
        segs = _transcribe_chunk_verbose(opus, task, language)
        for seg in segs:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segs)
    return segments_to_srt(all_segments)


def call_transcription(pcm_bytes: bytes, task: str, language: str | None) -> str:
    """
    Transcribe audio and return an SRT string.

    Always encodes the full PCM to Opus first so we can check the real file
    size before deciding whether to chunk. This avoids estimating from PCM
    duration and then hitting a 413 on the first API call.

      1. Encode to Opus  — fast, local. Check the actual output size.
      2. Fits?           — send as a single file.
      3. Too large?      — split the original PCM into N chunks (sample-accurate
                           offsets), encode each chunk, transcribe and merge.
      4. 413 fallback    — if the provider still rejects despite the size check
                           passing (lower actual limit than MAX_UPLOAD_MB), split
                           and retry.
    """
    # ── 1. Encode to Opus and check actual size ──────────────────────────────
    opus      = encode_pcm_to_opus(pcm_bytes)
    opus_size = opus.getbuffer().nbytes
    print(
        f"Encoded to Opus: {opus_size / 1024 / 1024:.1f} MB "
        f"(from {len(pcm_bytes) / 1024 / 1024:.1f} MB PCM, "
        f"{len(pcm_bytes) / opus_size:.0f}:1 compression)"
    )

    # ── 2. Single-file path ──────────────────────────────────────────────────
    if opus_size <= MAX_UPLOAD_BYTES:
        try:
            return _transcribe_single(opus, task, language)
        except Exception as exc:
            if not is_too_large_error(exc):
                raise
            # Provider's actual limit is lower than MAX_UPLOAD_BYTES.
            print(
                f"Provider returned 413 despite {opus_size / 1024 / 1024:.1f} MB Opus file. "
                f"Consider lowering MAX_UPLOAD_MB (currently {MAX_UPLOAD_BYTES // 1024 // 1024}). "
                f"Splitting into chunks and retrying."
            )
            # num_chunks based on actual size so each chunk is safely under the limit
            num_chunks = max(2, math.ceil(opus_size / MAX_UPLOAD_BYTES) + 1)
            return _transcribe_pcm_chunks(
                split_pcm_into_chunks(pcm_bytes, num_chunks), task, language
            )

    # ── 3. Pre-split path (actual Opus too large) ────────────────────────────
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
        f"using model: {whisper_model}, "
        f"opus: {OPUS_BITRATE_BPS // 1000} kbps / {OPUS_APPLICATION} (OPUS_BITRATE_KBPS), "
        f"max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB (MAX_UPLOAD_MB)"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
