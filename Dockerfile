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

# HuggingFace token for gated models (Flux Schnell)
# Pass as build arg: docker build --build-arg HF_TOKEN=hf_xxx ...
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Pre-download models during build for faster cold starts
# Requires HF_TOKEN for gated models
RUN if [ -n "$HF_TOKEN" ]; then \
        python -c "from src.models.loader import preload_models; preload_models()"; \
    else \
        echo "HF_TOKEN not provided, skipping model preload (will download at runtime)"; \
    fi

CMD ["python", "-u", "src/handler.py"]
