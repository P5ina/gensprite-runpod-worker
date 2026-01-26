"""
Model preloading for faster cold starts.
Downloads and caches models during Docker build.
"""

import os
import torch

# Cache directory for models
MODEL_CACHE = os.environ.get("HF_HOME", "/app/models")


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


def preload_models():
    """
    Preload all models during Docker build.
    This downloads models to the cache directory.
    """
    print("Preloading models...")

    # Flux Schnell for sprite generation
    print("Loading Flux Schnell...")
    from diffusers import FluxPipeline
    FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    )

    # SDXL for texture generation
    print("Loading SDXL...")
    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=MODEL_CACHE,
    )

    # Rembg model (u2net)
    print("Loading rembg model...")
    from rembg import new_session
    new_session("u2net")

    print("All models preloaded!")


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
            torch_dtype=get_dtype(),
            cache_dir=MODEL_CACHE,
        )
        _sprite_pipeline.to(get_device())
        # Enable memory optimizations
        if torch.cuda.is_available():
            _sprite_pipeline.enable_model_cpu_offload()
    return _sprite_pipeline


def get_texture_pipeline():
    """Get or create the SDXL pipeline for texture generation."""
    global _texture_pipeline
    if _texture_pipeline is None:
        from diffusers import StableDiffusionXLPipeline
        _texture_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=get_dtype(),
            variant="fp16",
            cache_dir=MODEL_CACHE,
        )
        _texture_pipeline.to(get_device())
        # Enable memory optimizations
        if torch.cuda.is_available():
            _texture_pipeline.enable_model_cpu_offload()
    return _texture_pipeline


def get_rembg_session():
    """Get or create the rembg session for background removal."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


if __name__ == "__main__":
    preload_models()
