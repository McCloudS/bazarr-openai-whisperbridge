# bazarr-openai-whisperbridge

A bridge that lets Bazarr's Whisper provider work with any OpenAI-compatible transcription endpoint — including OpenAI, Groq, and others.

> **Prefer fully self-hosted?** Check out [Subgen](https://github.com/McCloudS/subgen), which runs Whisper locally on your own hardware with integrations to Jellyfin, Plex, Emby, Tautulli, or Bazarr.

---

## Setup

Configure in Bazarr exactly as you would a regular Whisper provider, pointing it at this bridge's address and port (default 9000).

### Docker

```bash
docker pull mccloud/bazarr-openai-whisperbridge
```

```yaml
services:
  bazarr-openai-whisperbridge:
    image: mccloud/bazarr-openai-whisperbridge
    environment:
      - OPENAI_API_KEY=your_key_here
      - OPENAI_BASE_URL=https://api.groq.com/openai/v1  # omit for OpenAI
      - WHISPER_MODEL=whisper-large-v3-turbo
    ports:
      - 9000:9000
```

### Standalone

Requires Python 3.11+, ffmpeg in PATH, and the following packages:

```bash
pip install fastapi uvicorn python-multipart ffmpeg-python openai
```

Download `bazarr-openai-whisperbridge.py`, set your environment variables, and run:

```bash
python bazarr-openai-whisperbridge.py
```

---

## Using with Groq

Groq provides fast, free-tier Whisper inference as a drop-in replacement for OpenAI.

1. Generate an API key at https://console.groq.com/keys
2. Set these environment variables:

```yaml
- OPENAI_API_KEY=your_groq_key_here
- OPENAI_BASE_URL=https://api.groq.com/openai/v1
- WHISPER_MODEL=whisper-large-v3-turbo
```

**Model choice:**

| Model | Speed | Accuracy | Best for |
|---|---|---|---|
| `whisper-large-v3-turbo` | Fast | Good | Most content — recommended default |
| `whisper-large-v3` | Slower | Best | Difficult audio, heavy accents, multiple speakers |

> **Note:** `whisper-large-v3-turbo` does **not** support the `translate` task on Groq — you'll get a 400 error if Bazarr requests translation with that model. Switch to `whisper-large-v3` if you need translation.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** API key for your provider |
| `OPENAI_BASE_URL` | *(OpenAI)* | Custom provider endpoint. Omit to use OpenAI. Example: `https://api.groq.com/openai/v1` |
| `WHISPER_MODEL` | `whisper-1` | Model name passed to the provider. Use `whisper-large-v3-turbo` for Groq |
| `WHISPER_TRANSLATE_MODEL` | *(same as `WHISPER_MODEL`)* | Model used for translate tasks only. Set to `whisper-large-v3` to use the cheaper turbo model for transcription while keeping full v3 accuracy for translation |
| `FORCE_DETECTED_LANGUAGE_TO` | `en` | Language code returned when Bazarr calls `/detect-language`. Must be an ISO 639-1 code |
| `MAX_UPLOAD_MB` | `24` | File size limit in MB before audio is split into chunks, working around the 25 MB limit on OpenAI and Groq |
| `OPUS_BITRATE_KBPS` | `24` | Bitrate for Opus encoding before upload. 24 kbps keeps a 2-hour film under 24 MB with good quality. Increase for difficult audio |
| `MAX_LINE_LENGTH` | `42` | Maximum characters per subtitle line (Netflix guideline) |
| `GAP_SPLIT_SECS` | `0.4` | Silence gap in seconds that triggers a new subtitle. Prevents subtitles from displaying during pauses |
| `PHANTOM_START_MAX_SECS` | `1.0` | Filter phantom hallucination entries: drop any entry whose start time is within this many seconds of 0. Set to `0` to disable |
| `PHANTOM_DURATION_MAX_SECS` | `2.0` | Maximum duration (seconds) for a near-zero entry to be considered a phantom. Only takes effect alongside `PHANTOM_START_MAX_SECS` |

---

## Notes

- The provider must support `response_format=verbose_json` and `timestamp_granularities`
- Subtitles follow Netflix-style formatting: max 42 characters per line, 2 lines max, punctuation-aware line breaks, and gap-based silence suppression
- OpenAI's API has no language detection endpoint, so `/detect-language` returns `FORCE_DETECTED_LANGUAGE_TO` rather than analysing the audio
