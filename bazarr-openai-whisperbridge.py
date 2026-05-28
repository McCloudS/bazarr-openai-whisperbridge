version = '0.4'

import os
import io
import ffmpeg
import time
import threading
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union
from openai import OpenAI, BadRequestError
import uvicorn

# Check if running inside a Docker container
in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

# Initialize FastAPI & OpenAI Client
app = FastAPI()
client = OpenAI()

force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', 'en')
whisper_model = os.getenv('WHISPER_MODEL', 'whisper-1')

# Cached result of whether the configured provider supports response_format=srt.
# None  = not yet probed (first request will probe)
# True  = provider confirmed srt support
# False = provider rejected srt; use verbose_json + local conversion instead
#
# Protected by _provider_lock: the /asr handler runs in FastAPI's thread pool
# (sync def, not async def), so multiple requests can arrive concurrently.
# The lock ensures only one thread performs the probe; all others wait for the
# result rather than firing redundant probe requests.
_provider_supports_srt: bool | None = None
_provider_lock = threading.Lock()

def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert a float seconds value to an SRT timestamp string HH:MM:SS,mmm."""
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def verbose_json_to_srt(response) -> str:
    """
    Convert a verbose_json transcription/translation response to SRT format.
    Each segment has .start, .end, and .text attributes.
    """
    lines = []
    for i, segment in enumerate(response.segments, start=1):
        start_ts = seconds_to_srt_timestamp(segment.start)
        end_ts   = seconds_to_srt_timestamp(segment.end)
        text     = segment.text.strip()
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")
    return "\n".join(lines)


def is_format_rejection(exc: BadRequestError) -> bool:
    """
    Return True when a BadRequestError is specifically about response_format
    not being supported (e.g. Groq rejecting 'srt').
    """
    return "response_format" in str(exc).lower()


# ---------------------------------------------------------------------------
# Core transcription — probes once, then caches provider capability
# ---------------------------------------------------------------------------

def call_transcription(opus_data, task: str, language: str | None) -> str:
    """
    Request a transcription/translation from the provider and always return
    an SRT string.

    On the first call the function tries response_format="srt". If the provider
    rejects it with a format-related 400 the function transparently retries with
    response_format="verbose_json", converts the result to SRT locally, and
    caches the provider's capability so all subsequent calls skip the probe.

    Thread-safe: a lock ensures only one thread performs the probe; concurrent
    requests that arrive before the probe completes will wait briefly then
    proceed with the now-known format — they never fire redundant probes.
    """
    global _provider_supports_srt

    def _call(fmt: str):
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

    def _to_srt(response, fmt: str) -> str:
        if fmt == "verbose_json":
            return verbose_json_to_srt(response)
        return response  # srt: API returns a plain string

    # --- fast path: already know the answer, no lock needed ---
    if _provider_supports_srt is True:
        return _to_srt(_call("srt"), "srt")
    if _provider_supports_srt is False:
        return _to_srt(_call("verbose_json"), "verbose_json")

    # --- slow path: first request, probe under lock ---
    with _provider_lock:
        # Re-check inside the lock — another thread may have probed while we waited
        if _provider_supports_srt is True:
            return _to_srt(_call("srt"), "srt")
        if _provider_supports_srt is False:
            return _to_srt(_call("verbose_json"), "verbose_json")

        # We are the probing thread
        try:
            response = _call("srt")
            _provider_supports_srt = True
            print("Provider supports response_format=srt — caching for future requests.")
            return _to_srt(response, "srt")
        except BadRequestError as exc:
            if not is_format_rejection(exc):
                raise  # unrelated 400, propagate normally
            print(
                f"Provider rejected response_format=srt ({exc}). "
                "Retrying with verbose_json and converting locally — caching for future requests."
            )
            _provider_supports_srt = False
            return _to_srt(_call("verbose_json"), "verbose_json")

def convert_pcm_to_opus_in_memory(input_data) -> io.BytesIO:
    """
    Converts a raw PCM bytestream to Opus format and returns it as an in-memory BytesIO object.
    """
    try:
        input_data.seek(0)
        out, _ = (
            ffmpeg.input(
                "pipe:0",
                format="s16le",
                ar=16000,
                ac=1,
            )
            .output(
                "pipe:1",
                format="opus",
                ac=1,
                ar=16000,
                acodec="libopus",
                b="12k",
                application="voip",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, input=input_data.read())
        )
        opus_data = io.BytesIO(out)
        opus_data.seek(0)
        return opus_data

    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}") from e

@app.get("/status")
def status():
    """Endpoint to check the service status."""
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
    Endpoint to handle ASR requests from Bazarr.
    Converts the incoming PCM audio to Opus, sends it to the Whisper-compatible
    provider, and always returns SRT to Bazarr regardless of which response_format
    the provider natively supports.
    """
    try:
        start_time = time.time()
        opus_data = convert_pcm_to_opus_in_memory(audio_file.file)
        opus_data.name = "file.ogg"

        max_size_bytes = 25 * 1024 * 1024
        opus_size = opus_data.getbuffer().nbytes
        if opus_size > max_size_bytes and str(client.base_url).startswith('https://api.openai.com/'):
            raise ValueError(
                f"The Opus data exceeds the 25 MB limit for OpenAI "
                f"(size: {opus_size / (1024 * 1024):.2f} MB)."
            )

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
    print(f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} using model: {whisper_model}")
    uvicorn.run(app, host="0.0.0.0", port=9000)
