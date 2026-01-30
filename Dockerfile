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
ENV HF_HOME=/app/models
ENV U2NET_HOME=/app/models/u2net

# Download BiRefNet model for background removal (MIT license)
RUN python -c "from rembg import new_session; new_session('birefnet-general')"

# Download ungated diffusers models during build
# SDXL for texture generation (~6.5GB)
RUN python -c "from diffusers import StableDiffusionXLPipeline; \
    StableDiffusionXLPipeline.from_pretrained( \
        'stabilityai/stable-diffusion-xl-base-1.0', \
        torch_dtype='auto', \
        variant='fp16', \
        cache_dir='/app/models' \
    )"

# SV3D for rotation generation (~10GB)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download( \
        'chenguolin/sv3d-diffusers', \
        cache_dir='/app/models' \
    )"

# ControlNet Tile for rotation refinement (~2.5GB)
RUN python -c "from diffusers import ControlNetModel; \
    ControlNetModel.from_pretrained( \
        'TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic', \
        torch_dtype='auto', \
        cache_dir='/app/models' \
    )"

# RealESRGAN model for upscaling (download weights only)
RUN mkdir -p /app/models/realesrgan && \
    wget -O /app/models/realesrgan/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Copy source code
COPY src/ src/

# Gated models (Flux Schnell) downloaded at runtime with HF_TOKEN
# Set HF_TOKEN in RunPod env vars

CMD ["python", "-u", "src/handler.py"]
