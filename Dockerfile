# Use the official Python image as the base
FROM python:3.11-slim

# Set environment variables to ensure Python behaves as expected
ENV PYTHONDONTWRITEBYTECODE 1  # Prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED 1        # Ensure logs are shown in real time

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the application script into the container
COPY bazarr-openai-whisperbridge.py /app/

# Install Python dependencies directly with pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    ffmpeg-python \
    openai

# Command to run the app
CMD ["python", "bazarr-openai-whisperbridge.py"]
