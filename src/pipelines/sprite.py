"""
Sprite generation pipeline using Flux Schnell + rembg.
"""

import random
from typing import Optional, Callable, Awaitable
from PIL import Image
from rembg import remove

from models.loader import get_sprite_pipeline, get_rembg_session
from utils.blob import upload_image


async def generate_sprite(
    prompt: str,
    width: int,
    height: int,
    job_id: str,
    blob_token: str,
    seed: Optional[int] = None,
    on_progress: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> dict:
    """
    Generate a sprite with transparent background.

    Args:
        prompt: Text prompt describing the sprite
        width: Output width (default 1024)
        height: Output height (default 1024)
        job_id: Job ID for file naming
        blob_token: Vercel Blob token for uploads
        seed: Random seed for reproducibility
        on_progress: Async callback for progress updates

    Returns:
        Dict with result URLs:
        {
            "raw_url": "https://...",       # Original generation
            "processed_url": "https://...", # With transparent background
            "seed": 12345
        }
    """
    if on_progress:
        await on_progress(5, "Loading model...")

    # Get the Flux pipeline
    pipe = get_sprite_pipeline()

    # Set seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    import torch
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    if on_progress:
        await on_progress(10, "Generating sprite...")

    # Enhance prompt for better sprite generation
    enhanced_prompt = (
        f"{prompt}, game sprite, centered, isolated object, "
        "simple clean background, digital art, high quality"
    )

    # Generate image with Flux Schnell
    # Flux Schnell uses 4 steps with no CFG (guidance_scale=0.0)
    result = pipe(
        prompt=enhanced_prompt,
        width=width,
        height=height,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
    )

    raw_image = result.images[0]

    if on_progress:
        await on_progress(60, "Uploading raw image...")

    # Upload raw image
    raw_url = await upload_image(
        image=raw_image,
        path=f"sprites/{job_id}/raw.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(70, "Removing background...")

    # Remove background using rembg with optimized settings for sprites
    rembg_session = get_rembg_session()
    processed_image = remove(
        raw_image,
        session=rembg_session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=200,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=5,
    )

    # Ensure RGBA mode
    if processed_image.mode != "RGBA":
        processed_image = processed_image.convert("RGBA")

    if on_progress:
        await on_progress(90, "Uploading processed image...")

    # Upload processed image
    processed_url = await upload_image(
        image=processed_image,
        path=f"sprites/{job_id}/processed.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(100, "Completed")

    return {
        "raw_url": raw_url,
        "processed_url": processed_url,
        "seed": seed,
    }
