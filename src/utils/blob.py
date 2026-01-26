"""
Vercel Blob upload utilities.
"""

import os
import io
import httpx
from PIL import Image


BLOB_API_URL = "https://blob.vercel-storage.com"


async def upload_image(
    image: Image.Image,
    path: str,
    token: str,
    content_type: str = "image/png",
) -> str:
    """
    Upload an image to Vercel Blob storage.

    Args:
        image: PIL Image to upload
        path: Path/filename in blob storage (e.g., "sprites/abc123.png")
        token: Vercel Blob read-write token
        content_type: MIME type of the image

    Returns:
        Public URL of the uploaded blob
    """
    # Convert image to bytes
    buffer = io.BytesIO()

    if content_type == "image/png":
        image.save(buffer, format="PNG")
    elif content_type == "image/jpeg":
        image.save(buffer, format="JPEG", quality=95)
    elif content_type == "image/webp":
        image.save(buffer, format="WEBP", quality=95)
    else:
        image.save(buffer, format="PNG")
        content_type = "image/png"

    buffer.seek(0)
    image_bytes = buffer.getvalue()

    return await upload_bytes(image_bytes, path, token, content_type)


async def upload_bytes(
    data: bytes,
    path: str,
    token: str,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload raw bytes to Vercel Blob storage.

    Args:
        data: Bytes to upload
        path: Path/filename in blob storage
        token: Vercel Blob read-write token
        content_type: MIME type of the content

    Returns:
        Public URL of the uploaded blob
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{BLOB_API_URL}/{path}",
            content=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type,
                "x-api-version": "7",
                "x-content-type": content_type,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["url"]


async def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL.

    Args:
        url: URL of the image to download

    Returns:
        PIL Image
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=60.0)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
