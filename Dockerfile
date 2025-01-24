# Use the official Python image as the base
FROM python:3.11-alpine

# Set environment variables to ensure Python behaves as expected
ENV PYTHONDONTWRITEBYTECODE 1  # Prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED 1        # Ensure logs are shown in real time

# Install system dependencies
RUN apk add --no-cache \
    ffmpeg

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies directly with pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    ffmpeg-python \
    openai

# Copy the application script into the container
COPY bazarr-openai-whisperbridge.py /app/

# Command to run the app
CMD ["python", "bazarr-openai-whisperbridge.py"]
