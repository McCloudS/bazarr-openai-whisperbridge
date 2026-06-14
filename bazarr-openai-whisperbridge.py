version = '1.0'

import os
import io
import math
import re
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

# Netflix subtitle guidelines
MAX_LINE_LENGTH = int(os.getenv('MAX_LINE_LENGTH', '42'))  # chars per line
MAX_LINES       = 2                                         # lines per subtitle
# Start a new subtitle after a silence gap of this many seconds.
# Handles silence suppression — subtitle won't bleed into pauses.
GAP_SPLIT_SECS  = float(os.getenv('GAP_SPLIT_SECS', '0.4'))

# Phantom-segment filter (issue #3 — Groq/Whisper hallucination at t≈0)
# Drop any segment whose start is within this many seconds of 0, whose
# duration is shorter than the duration threshold, and whose text is a
# duplicate of a later entry.  Set PHANTOM_START_MAX_SECS=0 to disable.
PHANTOM_START_MAX_SECS    = float(os.getenv('PHANTOM_START_MAX_SECS',    '1.0'))
PHANTOM_DURATION_MAX_SECS = float(os.getenv('PHANTOM_DURATION_MAX_SECS', '2.0'))

PCM_SAMPLE_RATE      = 16000
PCM_NUM_CHANNELS     = 1
PCM_BITS_PER_SAMPLE  = 16
PCM_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

# Punctuation patterns for split decisions
_SENTENCE_END = re.compile(r'[.!?][\'")\]]*$')   # strong break after
_SOFT_BREAK   = re.compile(r'[,;:]$')             # weak break after
_CONJUNCTIONS = frozenset({
    'and', 'but', 'or', 'so', 'yet', 'for', 'nor',
    'as', 'if', 'when', 'then', 'because', 'although',
})


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis   = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis    = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n"
            f"{seconds_to_srt_timestamp(seg['start'])} --> {seconds_to_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(lines)


def filter_phantom_segments(segments: list[dict]) -> list[dict]:
    """
    Strip Whisper hallucination entries near timestamp 0 (issue #3).

    Groq (and occasionally OpenAI) emits two kinds of spurious near-zero
    segments:
      1. Duplicate text — same text appears again in a real later entry.
      2. Content-free — text contains no word characters (e.g. a lone ".").

    A segment is dropped when ALL of these hold:
      - start < PHANTOM_START_MAX_SECS
      - duration < PHANTOM_DURATION_MAX_SECS
      - text is a duplicate of a later entry OR contains no word characters

    A legitimate opening subtitle that is short but has unique, readable
    content is kept.  Set PHANTOM_START_MAX_SECS=0 to disable entirely.
    """
    if not segments or PHANTOM_START_MAX_SECS <= 0:
        return segments

    later_texts = {s["text"].strip().lower() for s in segments[1:]}
    result = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text_stripped = seg["text"].strip()
        is_phantom_text = (
            text_stripped.lower() in later_texts          # duplicate of later entry
            or not re.search(r'\w', text_stripped)        # no real word chars (e.g. ".")
        )
        if (
            seg["start"] < PHANTOM_START_MAX_SECS
            and duration < PHANTOM_DURATION_MAX_SECS
            and is_phantom_text
        ):
            print(
                f"Filtered phantom segment at "
                f"{seconds_to_srt_timestamp(seg['start'])}: "
                f'"{text_stripped}"'
            )
            continue
        result.append(seg)
    return result


# ---------------------------------------------------------------------------
# Netflix-style subtitle segmentation
# ---------------------------------------------------------------------------

def split_segments(words: list[dict]) -> list[dict]:
    """
    Convert a flat word list into subtitle segments following Netflix guidelines:
      - Max MAX_LINE_LENGTH characters per line
      - Max MAX_LINES lines per subtitle (2)
      - New subtitle after GAP_SPLIT_SECS silence (handles silence suppression)
      - New subtitle after sentence-ending punctuation
      - 2-line subtitles split at the most natural word boundary
      - Lines balanced in length; prefer breaking after punctuation,
        avoid breaking before conjunctions
    """
    if not words:
        return []

    segments: list[dict]       = []
    current: list[dict]        = []
    max_chars = MAX_LINE_LENGTH * MAX_LINES

    def flush():
        if current:
            segments.append(_format_subtitle(current))
            current.clear()

    for i, word in enumerate(words):
        # --- gap-based split (silence suppression) ---
        if current:
            gap = word["start"] - current[-1]["end"]
            if gap >= GAP_SPLIT_SECS:
                flush()

        # --- would exceed max subtitle length? ---
        candidate = " ".join(w["word"] for w in current) + (" " if current else "") + word["word"]
        if current and len(candidate) > max_chars:
            flush()

        current.append(word)

        # --- sentence-end split (only if subtitle is long enough to be worth it) ---
        text_so_far = " ".join(w["word"] for w in current)
        if _SENTENCE_END.search(word["word"]) and len(text_so_far) >= MAX_LINE_LENGTH // 2:
            flush()

    flush()
    return segments


def _format_subtitle(words: list[dict]) -> dict:
    """
    Format a word group as a 1- or 2-line subtitle dict.
    For 2-line subtitles, find the most natural split point:
      - Prefer after punctuation near the middle
      - Avoid breaking before conjunctions
      - Aim for balanced line lengths
    """
    text = " ".join(w["word"] for w in words)
    start = words[0]["start"]
    end   = words[-1]["end"]

    if len(text) <= MAX_LINE_LENGTH:
        return {"start": start, "end": end, "text": text}

    # Find best split point for 2 lines
    best_idx   = _find_line_split(words)
    line1 = " ".join(w["word"] for w in words[:best_idx])
    line2 = " ".join(w["word"] for w in words[best_idx:])

    # Safety: if either line still exceeds the limit, fall back to single block
    if len(line1) > MAX_LINE_LENGTH or len(line2) > MAX_LINE_LENGTH:
        return {"start": start, "end": end, "text": text}

    return {"start": start, "end": end, "text": f"{line1}\n{line2}"}


def _find_line_split(words: list[dict]) -> int:
    """
    Return the word index at which to split words into two balanced lines.

    Scoring (lower = better split):
      - Penalise imbalance between line lengths
      - Reward splitting after sentence-end or soft punctuation
      - Penalise splitting before a conjunction
    """
    texts = [w["word"] for w in words]
    n     = len(texts)

    best_idx   = max(1, n // 2)
    best_score = float("inf")

    for i in range(1, n):
        line1 = " ".join(texts[:i])
        line2 = " ".join(texts[i:])

        if len(line1) > MAX_LINE_LENGTH or len(line2) > MAX_LINE_LENGTH:
            continue

        balance      = abs(len(line1) - len(line2))
        punct_bonus  = -8 if _SENTENCE_END.search(texts[i - 1]) else \
                       -4 if _SOFT_BREAK.search(texts[i - 1]) else 0
        conj_penalty =  6 if texts[i].lower().rstrip(".,!?;:") in _CONJUNCTIONS else 0

        score = balance + punct_bonus + conj_penalty
        if score < best_score:
            best_score = score
            best_idx   = i

    return best_idx


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def verbose_json_to_segments(response) -> list[dict]:
    """
    Extract and segment the API response into subtitle-ready dicts.

    Groq returns words at the top level; OpenAI nests them in segments.
    Either way we pass the word list through split_segments() so line
    length, gap splitting, and punctuation-aware breaks are applied uniformly.
    """
    # --- Groq: top-level words ---
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
        print(f"Using {len(top_words)} top-level words.")
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
    all_segments: list[dict] = []
    for idx, (pcm_chunk, start_offset) in enumerate(pcm_chunks, start=1):
        print(f"Transcribing chunk {idx}/{len(pcm_chunks)} (offset={start_offset:.3f}s) ...")
        # Filter per-chunk before adding the offset so that a hallucinated
        # t=0 phantom on chunk N (which would shift to the chunk boundary
        # timestamp) is caught while it still looks like t=0.
        segs = filter_phantom_segments(_call_api(encode_pcm_to_opus(pcm_chunk), task, language))
        for seg in segs:
            seg["start"] += start_offset
            seg["end"]   += start_offset
        all_segments.extend(segs)
    return segments_to_srt(filter_phantom_segments(all_segments))


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
            return segments_to_srt(filter_phantom_segments(_call_api(opus, task, language)))
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
        f"max line: {MAX_LINE_LENGTH} chars | "
        f"gap split: {GAP_SPLIT_SECS}s"
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)
