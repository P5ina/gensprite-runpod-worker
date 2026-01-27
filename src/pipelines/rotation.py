"""
Rotation pipeline using SV3D (Stable Video 3D) for multi-view generation.
Generates 21 orbital frames and extracts 8 directions.
"""

import os
import math
from typing import Optional, Callable, Awaitable
from PIL import Image
import numpy as np
import torch
from rembg import remove

from models.loader import get_rembg_session, BAKED_MODEL_CACHE
from utils.blob import upload_image, download_image


# SV3D generates 21 frames
# Extract 8 frames at these indices (matching ComfyUI workflow)
DIRECTION_INDICES = {
    "S": 0,      # Front (0°)
    "SW": 3,     # ~51°
    "W": 5,      # ~86°
    "NW": 8,     # ~137°
    "N": 10,     # ~171° (back)
    "NE": 13,    # ~223°
    "E": 15,     # ~257°
    "SE": 18,    # ~309°
}


def preprocess_image(image: Image.Image, size: int = 576) -> Image.Image:
    """
    Preprocess input image for SV3D.
    - Remove background if needed
    - Center the object
    - Resize to target size
    - Add white background
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
        # No content found, return white image
        return Image.new("RGB", (size, size), (255, 255, 255))

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Crop to content
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]

    # Create square canvas with padding
    content_h, content_w = cropped.shape[:2]
    max_dim = max(content_h, content_w)
    padding = int(max_dim * 0.1)
    canvas_size = max_dim + padding * 2

    # Create white canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    # Center content on canvas
    y_offset = (canvas_size - content_h) // 2
    x_offset = (canvas_size - content_w) // 2

    # Composite over white background
    alpha_factor = cropped[:, :, 3:4] / 255.0
    for c in range(3):
        canvas[y_offset:y_offset+content_h, x_offset:x_offset+content_w, c] = (
            cropped[:, :, c] * alpha_factor[:, :, 0] +
            255 * (1 - alpha_factor[:, :, 0])
        ).astype(np.uint8)

    # Resize to target size
    result = Image.fromarray(canvas)
    result = result.resize((size, size), Image.Resampling.LANCZOS)

    return result


# Lazy-loaded pipeline
_sv3d_pipeline = None


def get_sv3d_pipeline():
    """Get or create the SV3D pipeline for rotation."""
    global _sv3d_pipeline
    if _sv3d_pipeline is not None:
        return _sv3d_pipeline

    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    # Import the custom SV3D components
    import sys
    sys.path.insert(0, "/app/src")
    from sv3d.pipeline import StableVideo3DDiffusionPipeline
    from sv3d.unet import SV3DUNetSpatioTemporalConditionModel

    model_path = os.path.join(BAKED_MODEL_CACHE, "models--chenguolin--sv3d-diffusers/snapshots")
    # Find the actual snapshot folder
    if os.path.exists(model_path):
        snapshots = os.listdir(model_path)
        if snapshots:
            model_path = os.path.join(model_path, snapshots[0])
    else:
        model_path = "chenguolin/sv3d-diffusers"

    # Load components
    # low_cpu_mem_usage=False prevents meta tensor errors when moving to GPU
    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_path, subfolder="feature_extractor"
    )

    _sv3d_pipeline = StableVideo3DDiffusionPipeline(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
    ).to("cuda")

    return _sv3d_pipeline


async def generate_rotation(
    input_image_url: str,
    elevation: int,
    job_id: str,
    blob_token: str,
    on_progress: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> dict:
    """
    Generate 8-directional rotation from a single input image using SV3D.
    """
    if on_progress:
        await on_progress(5, "Downloading input image...")

    # Download input image
    input_image = await download_image(input_image_url)

    if on_progress:
        await on_progress(10, "Preprocessing image...")

    # Preprocess (576x576, white background, centered)
    processed_input = preprocess_image(input_image, size=576)

    if on_progress:
        await on_progress(15, "Loading SV3D model...")

    pipe = get_sv3d_pipeline()

    if on_progress:
        await on_progress(20, "Generating orbital views...")

    # Calculate camera angles for SV3D
    num_frames = 21
    # Polar angle (elevation) - convert to radians
    polar_rad = [math.radians(90 - elevation)] * num_frames
    # Azimuth angles - full 360 rotation
    azimuths_rad = [math.radians(i * 360 / num_frames) for i in range(num_frames)]

    # Generate frames
    generator = torch.Generator(device="cuda").manual_seed(42)

    output = pipe(
        processed_input,
        height=576,
        width=576,
        num_frames=num_frames,
        polars_rad=polar_rad,
        azimuths_rad=azimuths_rad,
        decode_chunk_size=8,
        generator=generator,
        num_inference_steps=20,
    )

    frames = output.frames[0]  # List of PIL images

    if on_progress:
        await on_progress(60, "Processing rotations...")

    # Get rembg session
    rembg_session = get_rembg_session()

    # Extract and upload 8 directions
    results = {}
    direction_names = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]

    for i, direction in enumerate(direction_names):
        if on_progress:
            progress = 60 + int((i / len(direction_names)) * 35)
            await on_progress(progress, f"Processing {direction} direction...")

        frame_index = DIRECTION_INDICES[direction]
        frame = frames[frame_index]

        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # Remove background
        transparent_frame = remove(
            frame,
            session=rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
        )

        if transparent_frame.mode != "RGBA":
            transparent_frame = transparent_frame.convert("RGBA")

        # Upload
        url = await upload_image(
            image=transparent_frame,
            path=f"rotations/{job_id}/{direction.lower()}.png",
            token=blob_token,
        )

        results[f"rotation_{direction.lower()}"] = url

    if on_progress:
        await on_progress(100, "Completed")

    return results
