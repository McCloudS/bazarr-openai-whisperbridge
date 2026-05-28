version = '0.97-debug'

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

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

# Stable-ts regroup algorithm string.
# Controls how word-level timestamps are grouped into subtitle segments.
# Requires stable-ts-whisperless to be installed; skipped silently if not.
#
# Default mirrors subgen: clamp_max + split_by_length(84) + split_by_length(42).
# Set to blank to use stable-ts's own default (punctuation + gap based).
# See https://github.com/jianfch/stable-ts for the full string syntax.
REGROUP_ALGO = os.getenv('REGROUP', 'cm_sl=84_sl=42++++++1')

# ---------------------------------------------------------------------------
# Optional stable-ts import
# pip install stable-ts-whisperless (also requires torch CPU)
# ---------------------------------------------------------------------------
try:
    import stable_whisper
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
    """
    Extract segments from a verbose_json response, including word-level
    timestamps when the provider returns them (Groq/OpenAI with
    timestamp_granularities=["word"]).  Word data is used by stable-ts
    regroup for natural subtitle boundaries.
    """
    raw_segments = getattr(response, "segments", None) or []
    segments = []
    for seg in raw_segments:
        start = getattr(seg, "start", None)
        end   = getattr(seg, "end",   None)
        text  = (getattr(seg, "text", "") or "").strip()
        if start is None or end is None:
            continue
        s = {"start": start, "end": end, "text": text}
        words = getattr(seg, "words", None)
        if words:
            s["words"] = [
                {
                    "word":  getattr(w, "word", ""),
                    "start": w.start,
                    "end":   w.end,
                    "score": getattr(w, "probability", 1.0),
                }
                for w in words
                if getattr(w, "start", None) is not None
                and getattr(w, "end",   None) is not None
            ]
        segments.append(s)
    return segments


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
# stable-ts regrouping
# ---------------------------------------------------------------------------

def regroup_segments(segments: list[dict]) -> list[dict]:
    """
    Use stable-ts to regroup word-level timestamps into natural subtitle
    segments using the REGROUP algorithm string.

    Only runs when:
      - stable-ts-whisperless is installed
      - The segments contain word-level data (from timestamp_granularities)
      - REGROUP is not set to an empty string

    Falls back to the input segments on any error.
    """
    if not _stable_ts_available:
        return segments

    if not any(seg.get("words") for seg in segments):
        return segments

    if not REGROUP_ALGO:
        regroup_arg = True   # use stable-ts default
    else:
        regroup_arg = REGROUP_ALGO

    try:
        result = stable_whisper.WhisperResult({
            "segments": [
                {
                    "start": seg["start"],
                    "end":   seg["end"],
                    "text":  seg["text"],
                    "words": [
                        {
                            "word":        w.get("word", ""),
                            "start":       w.get("start"),
                            "end":         w.get("end"),
                            "probability": w.get("score", 1.0),
                        }
                        for w in (seg.get("words") or [])
                        if w.get("start") is not None and w.get("end") is not None
                    ],
                }
                for seg in segments
            ]
        })
        result.regroup(regroup_arg)
        return [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in result.segments
        ]
    except Exception as exc:
        print(f"stable-ts regroup failed ({exc}) — using original segments.")
        return [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments
        ]


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
    """
    Call the provider with verbose_json + word-level timestamps.
    Returns segments including word data when the provider supports it.
    """
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
    """Encode each PCM chunk, transcribe, regroup, apply offset, merge to SRT."""
    all_segments: list[dict] = []
    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        segs = _call_api(encode_pcm_to_opus(pcm_chunk), task, language)
        segs = regroup_segments(segs)
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
            segs = _call_api(opus, task, language)
            segs = regroup_segments(segs)
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

        return StreamingResponse(
            iter([srt_content or ""]),
            media_type="text/plain",
            headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    regroup_status = f"regroup: {REGROUP_ALGO or 'stable-ts default'}" if _stable_ts_available else "stable-ts not installed"
    print(
        f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} | "
        f"model: {whisper_model} | "
        f"opus: {OPUS_BITRATE_BPS // 1000} kbps | "
        f"max upload: {MAX_UPLOAD_BYTES // 1024 // 1024} MB | "
        f"{regroup_status}"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
