# Use official Python 3.12 slim image for minimal footprint
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system audio libraries (libsndfile for soundfile, ffmpeg for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels first for a lightweight container footprint (~800MB vs 4GB+ with CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code, checkpoints, and evaluation reports
COPY . .

# Default command runs inference on the bundled sample audio
CMD ["python", "inference.py", "--audio", "test_sample.wav"]
