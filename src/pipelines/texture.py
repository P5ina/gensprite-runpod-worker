"""
Texture generation pipeline using SDXL with seamless tiling + algorithmic PBR maps.
"""

import random
from typing import Optional, Callable, Awaitable
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

from models.loader import get_texture_pipeline
from utils.blob import upload_image


class SeamlessConv2d(nn.Module):
    """Wrapper for Conv2d that uses circular padding for seamless textures."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.padding = conv.padding

    def forward(self, x):
        # Apply circular padding manually
        if self.padding[0] > 0 or self.padding[1] > 0:
            # Circular pad: wrap around edges
            x = torch.nn.functional.pad(
                x,
                (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                mode='circular'
            )
        # Run conv with no padding (we already padded)
        return torch.nn.functional.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )


def patch_conv_for_seamless(model):
    """
    Recursively patch all Conv2d layers in a model to use circular padding.
    This makes the model generate seamlessly tileable textures.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.padding != (0, 0):
            setattr(model, name, SeamlessConv2d(module))
        else:
            patch_conv_for_seamless(module)


def unpatch_conv_for_seamless(model):
    """
    Recursively restore original Conv2d layers from SeamlessConv2d wrappers.
    """
    for name, module in model.named_children():
        if isinstance(module, SeamlessConv2d):
            setattr(model, name, module.conv)
        else:
            unpatch_conv_for_seamless(module)


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

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    if on_progress:
        await on_progress(10, "Preparing seamless generation...")

    # Patch UNet and VAE for seamless tiling (circular padding)
    patch_conv_for_seamless(pipe.unet)
    patch_conv_for_seamless(pipe.vae)

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

    if on_progress:
        await on_progress(15, "Generating seamless texture...")

    try:
        # Generate base color with SDXL (seamless due to circular padding)
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        )

        basecolor = result.images[0]
    finally:
        # Always restore original conv layers
        unpatch_conv_for_seamless(pipe.unet)
        unpatch_conv_for_seamless(pipe.vae)

    if on_progress:
        await on_progress(50, "Generating normal map...")

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
