FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache ffmpeg

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    ffmpeg-python \
    openai \
    numpy \
    stable-ts-whisperless

COPY bazarr-openai-whisperbridge.py /app/

CMD ["python", "bazarr-openai-whisperbridge.py"]
