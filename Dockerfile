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

# Pre-download models during build for faster cold starts
COPY src/models/loader.py src/models/loader.py
RUN python -c "from src.models.loader import preload_models; preload_models()"

# Copy source code
COPY src/ src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

CMD ["python", "-u", "src/handler.py"]
