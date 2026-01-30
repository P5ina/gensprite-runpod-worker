"""
Model preloading for faster cold starts.
Downloads and caches models during Docker build.
"""

import os
import torch

# Baked-in models (SDXL, SV3D, rembg) are in /app/models
BAKED_MODEL_CACHE = "/app/models"

# Legacy - keep for compatibility
MODEL_CACHE = BAKED_MODEL_CACHE

# U2NET_HOME is set via Dockerfile to /app/models/u2net (baked into image)


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
    Preload models during Docker build.
    """
    print("Preloading models...")

    # SDXL base for sprite and texture generation
    print("Loading SDXL base...")
    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=MODEL_CACHE,
    )

    # SDXL refiner for sprite refinement
    print("Loading SDXL refiner...")
    from diffusers import StableDiffusionXLImg2ImgPipeline
    StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=MODEL_CACHE,
    )

    # Rembg model (birefnet-general for better edge detection, MIT license)
    print("Loading rembg model...")
    from rembg import new_session
    new_session("birefnet-general")

    print("Models preloaded!")


# Lazy-loaded pipeline instances
_sprite_pipeline = None
_sprite_refiner = None
_texture_pipeline = None
_rembg_session = None
_tile_refiner_pipeline = None
_realesrgan_upscaler = None


def get_sprite_pipeline():
    """Get or create the SDXL base pipeline for sprite generation."""
    global _sprite_pipeline
    if _sprite_pipeline is None:
        from diffusers import StableDiffusionXLPipeline
        # SDXL is baked into the image
        _sprite_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=BAKED_MODEL_CACHE,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to("cuda")
    return _sprite_pipeline


def get_sprite_refiner():
    """Get or create the SDXL refiner pipeline for sprite generation."""
    global _sprite_refiner
    if _sprite_refiner is None:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        _sprite_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=BAKED_MODEL_CACHE,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to("cuda")
    return _sprite_refiner


def get_texture_pipeline():
    """Get or create the SDXL pipeline for texture generation."""
    global _texture_pipeline
    if _texture_pipeline is None:
        from diffusers import StableDiffusionXLPipeline
        # SDXL is baked into the image
        _texture_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=BAKED_MODEL_CACHE,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to("cuda")
    return _texture_pipeline


def get_rembg_session():
    """Get or create the rembg session for background removal."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        # birefnet-general has excellent edge detection (MIT license)
        _rembg_session = new_session("birefnet-general")
    return _rembg_session


def get_tile_refiner_pipeline():
    """Get or create the SDXL + ControlNet Tile pipeline for refinement."""
    global _tile_refiner_pipeline
    if _tile_refiner_pipeline is None:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        # Load ControlNet Tile model (xinsir version is in proper diffusers format)
        controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-tile-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir=BAKED_MODEL_CACHE,
        )

        # Create img2img pipeline with SDXL base + ControlNet for refinement
        _tile_refiner_pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=BAKED_MODEL_CACHE,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to("cuda")

    return _tile_refiner_pipeline


def get_realesrgan_upscaler():
    """Get or create the RealESRGAN upscaler using spandrel."""
    global _realesrgan_upscaler
    if _realesrgan_upscaler is None:
        import spandrel

        model_path = os.path.join(BAKED_MODEL_CACHE, "realesrgan", "RealESRGAN_x4plus.pth")

        # Spandrel auto-detects model architecture from weights
        _realesrgan_upscaler = spandrel.ModelLoader().load_from_file(model_path)
        _realesrgan_upscaler = _realesrgan_upscaler.to("cuda").eval()

    return _realesrgan_upscaler


if __name__ == "__main__":
    preload_models()
