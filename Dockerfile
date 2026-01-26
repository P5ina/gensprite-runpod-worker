FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/app/models

# Copy source code
COPY src/ src/

# Pre-download non-gated models during build
RUN python -c "from src.models.loader import preload_public_models; preload_public_models()"

# Note: Gated models (Flux) will be downloaded at runtime using HF_TOKEN env var
# Set HF_TOKEN in RunPod endpoint environment variables

CMD ["python", "-u", "src/handler.py"]
