"""
Texture generation pipeline using SDXL with seamless tiling + algorithmic PBR maps.
"""

import random
from typing import Optional, Callable, Awaitable
import numpy as np
from PIL import Image
import cv2

from models.loader import get_texture_pipeline
from utils.blob import upload_image


def make_seamless(image: Image.Image) -> Image.Image:
    """
    Make an image seamlessly tileable by blending edges.
    Uses cross-fading at borders.
    """
    img_array = np.array(image).astype(np.float32)
    h, w = img_array.shape[:2]

    # Blend width (percentage of image size)
    blend_size = min(w, h) // 8

    # Create horizontal blend
    for i in range(blend_size):
        alpha = i / blend_size
        # Left edge blends with right
        img_array[:, i] = (
            img_array[:, i] * alpha +
            img_array[:, w - blend_size + i] * (1 - alpha)
        )
        # Right edge blends with left
        img_array[:, w - 1 - i] = (
            img_array[:, w - 1 - i] * alpha +
            img_array[:, blend_size - 1 - i] * (1 - alpha)
        )

    # Create vertical blend
    for i in range(blend_size):
        alpha = i / blend_size
        # Top edge blends with bottom
        img_array[i, :] = (
            img_array[i, :] * alpha +
            img_array[h - blend_size + i, :] * (1 - alpha)
        )
        # Bottom edge blends with top
        img_array[h - 1 - i, :] = (
            img_array[h - 1 - i, :] * alpha +
            img_array[blend_size - 1 - i, :] * (1 - alpha)
        )

    return Image.fromarray(img_array.clip(0, 255).astype(np.uint8))


def generate_normal_map(image: Image.Image, strength: float = 2.0) -> Image.Image:
    """
    Generate a normal map from an image using Sobel operators.

    Args:
        image: Input image (base color)
        strength: Normal map strength/intensity

    Returns:
        RGB normal map image
    """
    # Convert to grayscale
    gray = np.array(image.convert("L")).astype(np.float32) / 255.0

    # Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Scale gradients
    sobel_x *= strength
    sobel_y *= strength

    # Create normal vectors (OpenGL convention: Y-up)
    normal_x = -sobel_x
    normal_y = -sobel_y
    normal_z = np.ones_like(gray)

    # Normalize
    magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= magnitude
    normal_y /= magnitude
    normal_z /= magnitude

    # Convert to RGB (0-255 range)
    # Normal map convention: R=X, G=Y, B=Z, mapped from [-1,1] to [0,255]
    r = ((normal_x + 1) / 2 * 255).astype(np.uint8)
    g = ((normal_y + 1) / 2 * 255).astype(np.uint8)
    b = ((normal_z + 1) / 2 * 255).astype(np.uint8)

    return Image.fromarray(np.stack([r, g, b], axis=-1))


def generate_height_map(image: Image.Image) -> Image.Image:
    """
    Generate a height/displacement map from the base color.
    Simply converts to grayscale with some contrast adjustment.
    """
    gray = image.convert("L")
    gray_array = np.array(gray).astype(np.float32)

    # Enhance contrast
    gray_array = (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min() + 1e-6)
    gray_array = (gray_array * 255).astype(np.uint8)

    return Image.fromarray(gray_array)


def generate_roughness_map(image: Image.Image, base_roughness: float = 0.5) -> Image.Image:
    """
    Generate a roughness map from the base color.
    Darker areas are considered smoother (lower roughness).

    Args:
        image: Input image
        base_roughness: Base roughness value (0-1)

    Returns:
        Grayscale roughness map
    """
    gray = np.array(image.convert("L")).astype(np.float32) / 255.0

    # Invert and adjust around base roughness
    # Lighter areas -> higher roughness, darker areas -> lower roughness
    roughness = base_roughness + (gray - 0.5) * 0.5
    roughness = np.clip(roughness, 0, 1)
    roughness = (roughness * 255).astype(np.uint8)

    return Image.fromarray(roughness)


def generate_metallic_map(image: Image.Image, threshold: float = 0.1) -> Image.Image:
    """
    Generate a metallic map. Most materials are non-metallic,
    so this returns a mostly dark map with slight variations.

    Args:
        image: Input image
        threshold: Max metallic value for non-metallic materials

    Returns:
        Grayscale metallic map (mostly black)
    """
    gray = np.array(image.convert("L")).astype(np.float32) / 255.0

    # Very subtle metallic variation based on brightness
    metallic = gray * threshold
    metallic = (metallic * 255).astype(np.uint8)

    return Image.fromarray(metallic)


async def generate_texture(
    prompt: str,
    job_id: str,
    blob_token: str,
    seed: Optional[int] = None,
    on_progress: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> dict:
    """
    Generate a PBR texture set.

    Args:
        prompt: Text prompt describing the texture
        job_id: Job ID for file naming
        blob_token: Vercel Blob token for uploads
        seed: Random seed for reproducibility
        on_progress: Async callback for progress updates

    Returns:
        Dict with result URLs:
        {
            "basecolor_url": "https://...",
            "normal_url": "https://...",
            "height_url": "https://...",
            "roughness_url": "https://...",
            "metallic_url": "https://...",
            "seed": 12345
        }
    """
    if on_progress:
        await on_progress(5, "Loading model...")

    # Get SDXL pipeline
    pipe = get_texture_pipeline()

    # Set seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    import torch
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    if on_progress:
        await on_progress(10, "Generating base texture...")

    # Enhance prompt for seamless texture generation
    enhanced_prompt = (
        f"{prompt}, seamless tileable texture, photorealistic material, "
        "high quality, detailed surface, even lighting, no shadows, "
        "flat texture view, PBR material"
    )

    negative_prompt = (
        "3d render, perspective, depth, shadows, uneven lighting, "
        "objects, people, text, watermark, logo, border, frame"
    )

    # Generate base color with SDXL
    result = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    )

    base_image = result.images[0]

    if on_progress:
        await on_progress(50, "Making texture seamless...")

    # Make seamless
    basecolor = make_seamless(base_image)

    if on_progress:
        await on_progress(55, "Generating normal map...")

    # Generate PBR maps algorithmically
    normal = generate_normal_map(basecolor)

    if on_progress:
        await on_progress(60, "Generating height map...")

    height = generate_height_map(basecolor)

    if on_progress:
        await on_progress(65, "Generating roughness map...")

    roughness = generate_roughness_map(basecolor)

    if on_progress:
        await on_progress(70, "Generating metallic map...")

    metallic = generate_metallic_map(basecolor)

    if on_progress:
        await on_progress(75, "Uploading textures...")

    # Upload all textures
    basecolor_url = await upload_image(
        image=basecolor,
        path=f"textures/{job_id}/basecolor.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(80, "Uploading normal map...")

    normal_url = await upload_image(
        image=normal,
        path=f"textures/{job_id}/normal.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(85, "Uploading height map...")

    height_url = await upload_image(
        image=height,
        path=f"textures/{job_id}/height.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(90, "Uploading roughness map...")

    roughness_url = await upload_image(
        image=roughness,
        path=f"textures/{job_id}/roughness.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(95, "Uploading metallic map...")

    metallic_url = await upload_image(
        image=metallic,
        path=f"textures/{job_id}/metallic.png",
        token=blob_token,
    )

    if on_progress:
        await on_progress(100, "Completed")

    return {
        "basecolor_url": basecolor_url,
        "normal_url": normal_url,
        "height_url": height_url,
        "roughness_url": roughness_url,
        "metallic_url": metallic_url,
        "seed": seed,
    }
