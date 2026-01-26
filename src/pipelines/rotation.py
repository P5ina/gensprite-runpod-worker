"""
Rotation pipeline using SVD (Stable Video Diffusion) for multi-view generation.
Generates orbital video frames and extracts 8 directions.
"""

from typing import Optional, Callable, Awaitable
from PIL import Image
import numpy as np
import torch
from rembg import remove

from models.loader import get_rembg_session, MODEL_CACHE, HF_TOKEN
from utils.blob import upload_image, download_image


# SVD generates 25 frames by default
# Map to 8 cardinal/ordinal directions
DIRECTION_INDICES = {
    "S": 0,      # Front (0°)
    "SW": 3,     # ~43°
    "W": 6,      # ~86°
    "NW": 9,     # ~130°
    "N": 12,     # ~173° (back)
    "NE": 15,    # ~216°
    "E": 18,     # ~259°
    "SE": 21,    # ~302°
}


def preprocess_input_image(image: Image.Image, size: int = 576) -> Image.Image:
    """
    Preprocess input image for SVD.
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
_svd_pipeline = None


def get_svd_pipeline():
    """Get or create the SVD pipeline for rotation."""
    global _svd_pipeline
    if _svd_pipeline is None:
        from diffusers import StableVideoDiffusionPipeline

        _svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=MODEL_CACHE,
            token=HF_TOKEN,
        ).to("cuda")

    return _svd_pipeline


async def generate_rotation(
    input_image_url: str,
    elevation: int,
    job_id: str,
    blob_token: str,
    on_progress: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> dict:
    """
    Generate 8-directional rotation from a single input image using SVD.
    """
    if on_progress:
        await on_progress(5, "Downloading input image...")

    # Download input image
    input_image = await download_image(input_image_url)

    if on_progress:
        await on_progress(10, "Preprocessing image...")

    # Preprocess (576x576, white background, centered)
    processed_input = preprocess_input_image(input_image, size=576)

    if on_progress:
        await on_progress(15, "Loading model...")

    pipe = get_svd_pipeline()

    if on_progress:
        await on_progress(20, "Generating orbital views...")

    # Generate video frames
    generator = torch.Generator(device="cuda").manual_seed(42)

    output = pipe(
        processed_input,
        num_frames=25,
        num_inference_steps=25,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=127,  # Controls motion amount
        noise_aug_strength=0.02,
    )

    frames = output.frames[0]  # List of PIL images

    if on_progress:
        await on_progress(60, "Processing rotations...")

    # Get rembg session
    rembg_session = get_rembg_session()

    # Extract and upload 8 directions
    results = {}
    direction_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

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
