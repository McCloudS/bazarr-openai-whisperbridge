# bazarr-openai-whisperbridge
A bridge to user Bazarr's Whisper provider with OpenAI formatted providers

### How to use:
## Docker: 
Download from Dockerhub @ `mccloud/bazarr-openai-whisperbridge` and set the following environment variables: `OPENAI_API_KEY` (Mandatory for all providers) & `OPENAI_BASE_URL` (optional if using OpenAI endpoint).
## Standalone:
Download the .py script, set the environment variables above, and have ffmpeg installed & `pip install fastapi uvicorn python-multipart ffmpeg-python openai`, run script.
