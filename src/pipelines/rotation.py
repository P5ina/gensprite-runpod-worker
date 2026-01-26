"""
Rotation pipeline using SyncDreamer for multi-view generation.
Generates 16 views and extracts 8 directions.
"""

import random
from typing import Optional, Callable, Awaitable, List
from PIL import Image
import numpy as np
from rembg import remove

from models.loader import get_rembg_session
from utils.blob import upload_image, download_image


# Direction mapping: indices from 16-view output to 8 cardinal/ordinal directions
# SyncDreamer generates 16 views evenly spaced around the object
# Index 0 is the front view, going counter-clockwise
DIRECTION_INDICES = {
    "N": 0,     # Front (0°)
    "NE": 2,    # 45°
    "E": 4,     # 90° (right)
    "SE": 6,    # 135°
    "S": 8,     # 180° (back)
    "SW": 10,   # 225°
    "W": 12,    # 270° (left)
    "NW": 14,   # 315°
}


def preprocess_input_image(image: Image.Image, size: int = 256) -> Image.Image:
    """
    Preprocess input image for SyncDreamer.
    - Remove background if needed
    - Center the object
    - Resize to target size
    - Add white background

    Args:
        image: Input PIL Image
        size: Target size (default 256 for SyncDreamer)

    Returns:
        Preprocessed image
    """
    # Convert to RGBA if needed
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Get alpha channel to check if background is already removed
    alpha = np.array(image.split()[-1])
    has_transparency = np.any(alpha < 250)

    if not has_transparency:
        # Remove background if image doesn't have transparency
        session = get_rembg_session()
        image = remove(image, session=session)

    # Convert to numpy for processing
    img_array = np.array(image)

    # Find bounding box of non-transparent pixels
    alpha = img_array[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No content found, return as-is
        return image.resize((size, size), Image.Resampling.LANCZOS)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Crop to content
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]

    # Create square canvas with padding
    content_h, content_w = cropped.shape[:2]
    max_dim = max(content_h, content_w)
    padding = int(max_dim * 0.1)  # 10% padding
    canvas_size = max_dim + padding * 2

    # Create white canvas with alpha
    canvas = np.ones((canvas_size, canvas_size, 4), dtype=np.uint8) * 255
    canvas[:, :, 3] = 255  # Fully opaque white background

    # Center content on canvas
    y_offset = (canvas_size - content_h) // 2
    x_offset = (canvas_size - content_w) // 2

    # Composite: place content over white background
    for c in range(3):
        alpha_factor = cropped[:, :, 3:4] / 255.0
        canvas[y_offset:y_offset+content_h, x_offset:x_offset+content_w, c] = (
            cropped[:, :, c] * alpha_factor[:, :, 0] +
            255 * (1 - alpha_factor[:, :, 0])
        ).astype(np.uint8)

    # Resize to target size
    result = Image.fromarray(canvas[:, :, :3])  # RGB only
    result = result.resize((size, size), Image.Resampling.LANCZOS)

    return result


async def generate_with_syncdreamer(input_image: Image.Image, elevation: int = 20) -> List[Image.Image]:
    """
    Generate multi-view images using SyncDreamer.

    Args:
        input_image: Preprocessed input image (256x256, white background)
        elevation: Camera elevation angle in degrees

    Returns:
        List of 16 PIL Images representing views around the object
    """
    import torch
    from diffusers import DiffusionPipeline

    # Load SyncDreamer pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "dylanebert/SyncDreamer",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Generate views
    # SyncDreamer expects elevation in radians for some versions
    result = pipe(
        input_image,
        num_inference_steps=50,
        guidance_scale=2.0,
        elevation=elevation,
    )

    return result.images


async def generate_rotation(
    input_image_url: str,
    elevation: int,
    job_id: str,
    blob_token: str,
    on_progress: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> dict:
    """
    Generate 8-directional rotation from a single input image.

    Args:
        input_image_url: URL of the input image
        elevation: Camera elevation angle (-90 to 90)
        job_id: Job ID for file naming
        blob_token: Vercel Blob token for uploads
        on_progress: Async callback for progress updates

    Returns:
        Dict with rotation URLs:
        {
            "rotation_n": "https://...",
            "rotation_ne": "https://...",
            "rotation_e": "https://...",
            "rotation_se": "https://...",
            "rotation_s": "https://...",
            "rotation_sw": "https://...",
            "rotation_w": "https://...",
            "rotation_nw": "https://...",
        }
    """
    if on_progress:
        await on_progress(5, "Downloading input image...")

    # Download input image
    input_image = await download_image(input_image_url)

    if on_progress:
        await on_progress(10, "Preprocessing image...")

    # Preprocess for SyncDreamer
    processed_input = preprocess_input_image(input_image, size=256)

    if on_progress:
        await on_progress(15, "Generating multi-view images...")

    # Generate 16 views with SyncDreamer
    views = await generate_with_syncdreamer(processed_input, elevation=elevation)

    if on_progress:
        await on_progress(60, "Processing rotations...")

    # Get rembg session for background removal
    rembg_session = get_rembg_session()

    # Extract and upload 8 directions
    results = {}
    direction_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    for i, direction in enumerate(direction_names):
        if on_progress:
            progress = 60 + int((i / len(direction_names)) * 35)
            await on_progress(progress, f"Processing {direction} direction...")

        # Get the view for this direction
        view_index = DIRECTION_INDICES[direction]
        view_image = views[view_index]

        # Remove background for transparency
        transparent_view = remove(
            view_image,
            session=rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
        )

        # Ensure RGBA
        if transparent_view.mode != "RGBA":
            transparent_view = transparent_view.convert("RGBA")

        # Upload
        url = await upload_image(
            image=transparent_view,
            path=f"rotations/{job_id}/{direction.lower()}.png",
            token=blob_token,
        )

        results[f"rotation_{direction.lower()}"] = url

    if on_progress:
        await on_progress(100, "Completed")

    return results
