"""
Model preloading for faster cold starts.
Downloads and caches models during Docker build.
"""

import os
import torch

# Cache directory for models - use network volume if available
MODEL_CACHE = os.environ.get("HF_HOME", "/runpod-volume/models")

# HuggingFace token for gated models (Flux Schnell)
HF_TOKEN = os.environ.get("HF_TOKEN")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype():
    """Get optimal dtype for the device."""
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def preload_public_models():
    """
    Preload non-gated models during Docker build.
    Gated models (Flux) will be downloaded at runtime with HF_TOKEN.
    """
    print("Preloading public models...")

    # SDXL for texture generation (public model)
    print("Loading SDXL...")
    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=MODEL_CACHE,
    )

    # Rembg model (isnet-general-use for better edge detection)
    print("Loading rembg model...")
    from rembg import new_session
    new_session("isnet-general-use")

    print("Public models preloaded!")
    print("Note: Flux Schnell (gated) will download at runtime with HF_TOKEN")


def preload_models():
    """
    Preload all models including gated ones.
    Requires HF_TOKEN environment variable.
    """
    print("Preloading all models...")

    # Flux Schnell for sprite generation (gated model, requires HF_TOKEN)
    print("Loading Flux Schnell...")
    from diffusers import FluxPipeline
    FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
        token=HF_TOKEN,
    )

    # Also preload public models
    preload_public_models()


# Lazy-loaded pipeline instances
_sprite_pipeline = None
_texture_pipeline = None
_rembg_session = None


def get_sprite_pipeline():
    """Get or create the Flux pipeline for sprite generation."""
    global _sprite_pipeline
    if _sprite_pipeline is None:
        from diffusers import FluxPipeline
        _sprite_pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            token=HF_TOKEN,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to("cuda")
    return _sprite_pipeline


def get_texture_pipeline():
    """Get or create the SDXL pipeline for texture generation."""
    global _texture_pipeline
    if _texture_pipeline is None:
        from diffusers import StableDiffusionXLPipeline
        _texture_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=MODEL_CACHE,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to("cuda")
    return _texture_pipeline


def get_rembg_session():
    """Get or create the rembg session for background removal."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        # isnet-general-use has better edge detection for game sprites
        _rembg_session = new_session("isnet-general-use")
    return _rembg_session


if __name__ == "__main__":
    preload_models()
