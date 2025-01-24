# bazarr-openai-whisperbridge
A bridge to user Bazarr's Whisper provider with OpenAI formatted providers

### How to use:
Configure in Bazarr the same as a regular Whisper provider.
## Docker: 
Download from Dockerhub @ `mccloud/bazarr-openai-whisperbridge` and set the following environment variables: `OPENAI_API_KEY` (Mandatory for all providers) & `OPENAI_BASE_URL` (optional if using OpenAI endpoint), and map your port (default 9000).
## Standalone:
Download the .py script, set the environment variables above, and have ffmpeg installed & `pip install fastapi uvicorn python-multipart ffmpeg-python openai`, run script.

### Variables:

| Variable              | Default Value | Description                                                                                                                                                                              |
|-----------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OPENAI_API_KEY         | ''        | Required for all providers |
| OPENAI_BASE_URL | '' | Optional if you want a custom provider, otherwise the OpenAI client defaults it to OpenAI's endpoint |
| FORCE_DETECTED_LANGUAGE_TO | 'en' | If detect_language is called from Bazarr's Whisper provider, it will return this language, must be a ISO 639-1 letter code |

# Caveats/Notes
* OpenAI's Whisper endpoint can only take 25mb files.  This attempts to convert the WAV to a more compressed OPUS file to combat that.  You will still run into this issue on large/long files (probably > 90 minutes).  This issue won't exist on other providers.

* OpenAI's endpoint does not have a detect language equivalent, so we have to force it to what we want or it will default to return English.
