FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Use only-if-needed to avoid reinstalling packages already in base image
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/runpod-volume/models

# Copy source code
COPY src/ src/

# Models will be downloaded at runtime and cached on network volume
# Set HF_TOKEN and HF_HOME=/runpod-volume/models in RunPod env vars

CMD ["python", "-u", "src/handler.py"]
