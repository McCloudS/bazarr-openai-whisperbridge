version = '0.93'

import os
import io
import math
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

MAX_UPLOAD_BYTES  = int(os.getenv('MAX_UPLOAD_MB', '24')) * 1024 * 1024
OPUS_BITRATE_BPS  = int(os.getenv('OPUS_BITRATE_KBPS', '24')) * 1000
OPUS_APPLICATION  = "audio"

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

# ---------------------------------------------------------------------------
# Optional stable-ts import
# If stable-ts and numpy are installed, segment boundaries are snapped to the
# nearest silence point in the audio after transcription, improving subtitle
# timing without any change to the API provider or model.
# Install: pip install stable-ts numpy
# ---------------------------------------------------------------------------
try:
    import stable_whisper
    import numpy as np
    _stable_ts_available = True
except ImportError:
    _stable_ts_available = False

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
# stable-ts refinement
# ---------------------------------------------------------------------------

def refine_segments(segments: list[dict], pcm_bytes: bytes) -> list[dict]:
    """
    Refine segment timestamps using stable-ts's transcribe_any() — the
    documented API for using stable-ts with any external ASR model or API.

    Rather than constructing silence arrays manually and calling suppress_silence,
    we pass stable-ts a mock inference function that returns our already-computed
    segments. stable-ts then:
      1. Analyses the raw audio itself to detect non-speech regions
      2. Calls our mock function (which returns the pre-computed segments)
      3. Applies its own silence suppression to adjust segment boundaries

    This is the intended usage pattern for external ASR — stable-ts does all
    the audio analysis, we just supply the transcription text and timestamps.

    Refinement is applied while timestamps are still relative to the chunk start
    (before the offset is added), so stable-ts is always analysing the correct
    audio window.

    Falls back to the original segments silently on any error.
    """
    if not _stable_ts_available or not segments:
        return segments

    try:
        audio = np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0

        precomputed = {
            "text": " ".join(s["text"] for s in segments),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"], "words": []}
                for s in segments
            ],
        }

        # stable-ts passes the numpy array directly (audio_type="numpy"); we ignore
        # the audio input and return our pre-computed API result.
        result = stable_whisper.transcribe_any(
            lambda *args, **kwargs: precomputed,
            audio,
            input_sr=PCM_SAMPLE_RATE,
            audio_type="numpy",
            suppress_silence=True,
        )

        return [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in result.segments
        ]
    except Exception as exc:
        print(f"stable-ts refinement failed ({exc}) — using unrefined segments.")
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
    """
    Split raw PCM into num_chunks equal pieces with sample-accurate offsets.
    Splitting happens on the PCM before encoding so timestamps are exact.
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

def _call_api(opus: io.BytesIO, task: str, language: str | None) -> list[dict]:
    """
    Call the provider with response_format=verbose_json and return segments.

    We always use verbose_json rather than srt because:
      - stable-ts needs structured segment data (start/end/text) to apply
        silence suppression — it cannot work from a raw SRT string.
      - Our own verbose_json → SRT conversion is faithful, so there is no
        quality difference vs the provider's native SRT output.
      - This removes the need for the srt-support probe, the provider cache,
        and the threading lock — the code is simpler and works with any
        OpenAI-compatible provider without a first-request round trip.
    """
    opus.seek(0)
    if task == "transcribe":
        response = client.audio.transcriptions.create(
            model=whisper_model, file=opus, response_format="verbose_json", language=language,
        )
    else:
        response = client.audio.translations.create(
            model=whisper_model, file=opus, response_format="verbose_json",
        )
    return verbose_json_to_segments(response)


def _transcribe_pcm_chunks(
    pcm_chunks: list[tuple[bytes, float]],
    task: str,
    language: str | None,
) -> str:
    """
    Encode each PCM chunk to Opus, transcribe, optionally refine with stable-ts,
    apply the sample-accurate time offset, then merge all segments into one SRT.

    stable-ts refinement runs against each chunk's own PCM audio while the
    timestamps are still chunk-relative (before the offset is added), so
    silence detection is always looking at the correct audio window.
    """
    all_segments: list[dict] = []

    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        opus = encode_pcm_to_opus(pcm_chunk)
        segs = _call_api(opus, task, language)

        # Refine before offsetting — stable-ts analyses the chunk's audio and
        # timestamps are still relative to the chunk start at this point.
        segs = refine_segments(segs, pcm_chunk)

        for seg in segs:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segs)

    return segments_to_srt(all_segments)


def call_transcription(pcm_bytes: bytes, task: str, language: str | None) -> str:
    """
    Transcribe audio and return an SRT string.

      1. Encode to Opus  — check the actual output size.
      2. Fits?           — transcribe as a single file, refine with stable-ts.
      3. Too large?      — split PCM into N chunks, encode each, transcribe,
                           refine, offset, merge.
      4. 413 fallback    — force-split and retry if the provider rejects despite
                           the size check passing.
    """
    # ── 1. Encode and check actual size ─────────────────────────────────────
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
            segs = _call_api(opus, task, language)
            segs = refine_segments(segs, pcm_bytes)
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

    # ── 3. Pre-split path ────────────────────────────────────────────────────
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
    stable_ts_status = f"stable-ts active" if _stable_ts_available else "stable-ts not installed"
    print(
        f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} | "
        f"model: {whisper_model} | "
        f"opus: {OPUS_BITRATE_BPS // 1000} kbps | "
        f"max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB | "
        f"{stable_ts_status}"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
