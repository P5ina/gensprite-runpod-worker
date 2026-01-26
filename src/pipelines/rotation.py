"""
Rotation pipeline - DISABLED FOR MVP.
Multi-view generation requires a compatible model (Zero123++, etc.)
TODO: Implement with Zero123++ or similar model.
"""

from typing import Optional, Callable, Awaitable


async def generate_rotation(
    input_image_url: str,
    elevation: int,
    job_id: str,
    blob_token: str,
    on_progress: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> dict:
    """
    Generate 8-directional rotation from a single input image.

    NOTE: This feature is disabled for MVP. The SyncDreamer model is not
    compatible with diffusers. Will be implemented with Zero123++ later.
    """
    raise NotImplementedError(
        "Rotation feature is temporarily disabled. "
        "Multi-view generation will be available in a future update."
    )
