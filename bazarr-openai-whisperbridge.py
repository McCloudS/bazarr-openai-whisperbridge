version = '0.3'

import os
import io
import ffmpeg
import time
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Union
from openai import OpenAI
import logging
import sys
import uvicorn

# Check if running inside a Docker container
in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

# Initialize FastAPI & OpenAI Client
app = FastAPI()
client = OpenAI()

force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', 'en')
whisper_model = os.getenv('WHISPER_MODEL', 'whisper-1')

def convert_pcm_to_opus_in_memory(input_data) -> io.BytesIO:
    """
    Converts a raw PCM bytestream to Opus format and returns it as an in-memory BytesIO object.

    Parameters:
        input_data (BytesIO): The raw PCM audio data, read from an UploadFile (e.g., input_data.file.read()).

    Returns:
        io.BytesIO: The converted Opus audio as an in-memory file-like object.
    """
    try:
        # Seek to the start of the input data, if not already
        input_data.seek(0)

        # Use FFmpeg to process the raw PCM data and output it to stdout
        out, _ = (
            ffmpeg.input(
                "pipe:0",  # Input from stdin
                format="s16le",  # Raw PCM format
                ar=16000,        # Sample rate
                ac=1             # Mono audio
            )
            .output(
                "pipe:1",        # Output to stdout
                format="opus",   # Output format
                ac=1,            # Mono audio
                ar=16000,        # Sample rate
                acodec="libopus",# Opus codec
                b="12k",         # Bitrate
                application="voip" # Opus application mode
            )
            .overwrite_output()  # Overwrite existing files (not applicable here)
            .run(capture_stdout=True, capture_stderr=True, input=input_data.read())
        )

        # Wrap the output in a BytesIO object
        opus_data = io.BytesIO(out)
        opus_data.seek(0)  # Reset the cursor to the beginning of the stream
        
        return opus_data

    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}") from e

@app.get("/status")
def status():
    """
    Endpoint to check the service status.
    """
    return {"version": f"Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version}"}

@app.post("/detect-language")
def detect_language():
    print(f"Forced detected language to {force_detected_language_to}")
    return {"detected_language": f"Forced to {force_detected_language_to} from WhisperBridge", "language_code": force_detected_language_to}
    
@app.post("/asr")
async def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
):
    """
    Endpoint to handle ASR (Automatic Speech Recognition) requests from Bazarr.
    Accepts an audio file, converts it in memory to Opus format, and sends it to OpenAI Whisper API.
    """
    try:
        start_time = time.time()
        opus_data = convert_pcm_to_opus_in_memory(audio_file.file)
        opus_data.name = "file.ogg"
        
        max_size_bytes = 25 * 1024 * 1024  # 25 MB in bytes

        # Check the size of opus_data
        opus_size = opus_data.getbuffer().nbytes
        if opus_size > max_size_bytes and client.base_url == 'https://api.openai.com/v1/':
            print(f"The Opus data exceeds the 25 MB limit for OpenAI (size: {opus_size / (1024 * 1024):.2f} MB).")
            raise ValueError(f"The Opus data exceeds the 25 MB limit for OpenAI (size: {opus_size / (1024 * 1024):.2f} MB).")
        
        # Send the Opus data to OpenAI Whisper API
        if task == "transcribe":
            print("Got a transcribe task from Bazarr")
            response = client.audio.transcriptions.create(
                model=whisper_model,
                file=opus_data,
                response_format="srt",
                language=language,
            )
        else:
            print("Got a translate task from Bazarr")
            response = client.audio.translations.create(
                model=whisper_model,
                file=opus_data,
                response_format="srt",
            )
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        print(f"Transcription of '{video_file}' from Bazarr complete, it took {minutes} minutes and {seconds} seconds to complete." if video_file 
            else f"Transcription complete, it took {minutes} minutes and {seconds} seconds to complete.")

        if response:
            return StreamingResponse(
                response,
                media_type="text/plain",
                headers={"Source": "Transcribed using Bazarr to OpenAI Whisper Bridge!"},
            )
        else:
            return

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"Running Bazarr to OpenAI Whisper Bridge ({docker_status}) v{version} using model: {whisper_model}") 
    uvicorn.run(app, host="0.0.0.0", port=9000)
