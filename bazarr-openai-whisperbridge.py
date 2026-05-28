version = '0.96'

import os
import io
import math
import threading
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

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

# ---------------------------------------------------------------------------
# Optional WhisperX alignment
# If whisperx is installed, word-level forced alignment is applied after the
# API transcription, giving more accurate subtitle boundaries.
# Install: pip install whisperx (also requires torch)
# Models are downloaded on first use per language and cached in
# ~/.cache/torch — mount a volume there to persist across container restarts.
# ---------------------------------------------------------------------------
try:
    import whisperx
    import numpy as np
    _whisperx_available = True
except ImportError:
    _whisperx_available = False

# Per-language alignment model cache.
# Models are loaded once and reused across requests.
# Protected by _align_lock so concurrent requests don't trigger duplicate loads.
_align_models: dict = {}
_align_lock = threading.Lock()


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


# ---------------------------------------------------------------------------
# WhisperX alignment
# ---------------------------------------------------------------------------

def _get_align_model(language_code: str):
    """
    Load and cache the wav2vec2 alignment model for a given language.
    The first load downloads the model (~200-400 MB depending on language).
    Subsequent calls return the cached model instantly.
    """
    if language_code not in _align_models:
        with _align_lock:
            if language_code not in _align_models:
                print(f"Loading WhisperX alignment model for '{language_code}' ...")
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device="cpu",
                )
                _align_models[language_code] = (model_a, metadata)
                print(f"Alignment model for '{language_code}' loaded and cached.")
    return _align_models[language_code]


def align_segments(
    segments: list[dict],
    pcm_bytes: bytes,
    language_code: str,
) -> list[dict]:
    """
    Apply WhisperX forced alignment to refine segment timestamps.

    Uses a language-specific wav2vec2 phoneme model to match each word in the
    transcript to its exact position in the audio. This is fundamentally more
    accurate than silence detection — it actually reads the speech, not just
    the quiet gaps.

    The aligned segment boundaries (start of first word, end of last word) are
    used for SRT output. Falls back to the original segments on any error,
    including unsupported languages.
    """
    if not _whisperx_available or not segments:
        return segments

    try:
        model_a, metadata = _get_align_model(language_code)
        audio = np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0
        result = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            device="cpu",
            return_char_alignments=False,
        )

        refined = []
        for seg in result["segments"]:
            words = seg.get("words", [])
            # Use the aligned word boundaries for tighter start/end times.
            # Fall back to the original segment times if words are missing.
            start = words[0]["start"]  if words and "start" in words[0]  else seg["start"]
            end   = words[-1]["end"]   if words and "end"   in words[-1] else seg["end"]
            refined.append({
                "start": start,
                "end":   end,
                "text":  seg["text"].strip(),
            })
        return refined

    except Exception as exc:
        print(f"WhisperX alignment failed ({exc}) — using unaligned segments.")
        return segments


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

def _call_api(opus: io.BytesIO, task: str, language: str | None) -> tuple[list[dict], str]:
    """
    Call the provider with verbose_json.
    Returns (segments, detected_language) — the detected language is used by
    the WhisperX alignment model loader to pick the correct wav2vec2 model.
    For translations the output is always English, so we return 'en'.
    """
    opus.seek(0)
    if task == "transcribe":
        response = client.audio.transcriptions.create(
            model=whisper_model, file=opus, response_format="verbose_json", language=language,
        )
        detected_language = getattr(response, "language", None) or language or "en"
    else:
        response = client.audio.translations.create(
            model=whisper_model, file=opus, response_format="verbose_json",
        )
        detected_language = "en"  # translations are always English

    return verbose_json_to_segments(response), detected_language


def _transcribe_pcm_chunks(
    pcm_chunks: list[tuple[bytes, float]],
    task: str,
    language: str | None,
) -> str:
    """
    Encode each PCM chunk to Opus, transcribe, align (if WhisperX available),
    apply sample-accurate offset, and merge into one SRT.

    Alignment runs against each chunk's own PCM audio while timestamps are
    still chunk-relative (before the offset is added), so the wav2vec2 model
    is always working against the correct audio window.
    """
    all_segments: list[dict] = []
    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        segs, detected_language = _call_api(encode_pcm_to_opus(pcm_chunk), task, language)
        segs = align_segments(segs, pcm_chunk, detected_language)
        for seg in segs:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segs)
    return segments_to_srt(all_segments)


def call_transcription(pcm_bytes: bytes, task: str, language: str | None) -> str:
    """
    Transcribe audio and return an SRT string.

      1. Encode to Opus  — check the actual output size.
      2. Fits?           — single API call, then align.
      3. Too large?      — split PCM, encode each chunk, transcribe, align,
                           apply sample-accurate offset, merge.
      4. 413 fallback    — force-split and retry.
    """
    opus      = encode_pcm_to_opus(pcm_bytes)
    opus_size = opus.getbuffer().nbytes
    print(
        f"Encoded to Opus: {opus_size / 1024 / 1024:.1f} MB "
        f"(from {len(pcm_bytes) / 1024 / 1024:.1f} MB PCM, "
        f"{len(pcm_bytes) / opus_size:.0f}:1 compression)"
    )

    if opus_size <= MAX_UPLOAD_BYTES:
        try:
            segs, detected_language = _call_api(opus, task, language)
            segs = align_segments(segs, pcm_bytes, detected_language)
            return segments_to_srt(segs)
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

        if srt_content:
            return StreamingResponse(
                iter([srt_content]),
                media_type="text/plain",
                headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    whisperx_status = "whisperx active" if _whisperx_available else "whisperx not installed"
    print(
        f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} | "
        f"model: {whisper_model} | "
        f"opus: {OPUS_BITRATE_BPS // 1000} kbps | "
        f"max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB | "
        f"{whisperx_status}"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
