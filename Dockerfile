FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart ffmpeg-python openai

COPY bazarr-openai-whisperbridge.py /app/
CMD ["python", "bazarr-openai-whisperbridge.py"]
