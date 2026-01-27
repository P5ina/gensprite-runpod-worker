"""
Main RunPod handler for GenSprite worker.
Routes jobs to appropriate pipelines and returns results.
"""

import os
import runpod

# Import pipelines
from pipelines.sprite import generate_sprite
from pipelines.texture import generate_texture
from pipelines.rotation import generate_rotation


# Environment variables
BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "")


async def handler(job: dict) -> dict:
    """
    Async handler for RunPod serverless.
    Returns a result dict on success or an error dict on failure.

    Expected job input format:
    {
        "job_id": "abc123",           # ID in main app's database
        "job_type": "sprite",         # sprite | texture | rotation
        "blob_token": "...",          # Vercel Blob token

        # Job-specific parameters:
        # For sprite:
        "prompt": "a cute robot",
        "width": 1024,
        "height": 1024,
        "seed": 12345,  # optional

        # For texture:
        "prompt": "brick wall texture",
        "seed": 12345,  # optional

        # For rotation:
        "input_image_url": "https://...",
        "elevation": 20,
    }
    """
    job_input = job.get("input", {})

    # Extract common parameters
    job_id = job_input.get("job_id")
    job_type = job_input.get("job_type")
    blob_token = job_input.get("blob_token", BLOB_TOKEN)

    if not job_id:
        return {"status": "failed", "error": "Missing job_id"}
    if not job_type:
        return {"status": "failed", "error": "Missing job_type"}
    if not blob_token:
        return {"status": "failed", "error": "Missing blob_token"}

    # No-op progress callback (pipelines still call it but nothing consumes it)
    async def on_progress(progress: int, stage: str):
        pass

    try:
        if job_type == "sprite":
            result = await generate_sprite(
                prompt=job_input.get("prompt", ""),
                width=job_input.get("width", 1024),
                height=job_input.get("height", 1024),
                seed=job_input.get("seed"),
                job_id=job_id,
                blob_token=blob_token,
                on_progress=on_progress,
            )
        elif job_type == "texture":
            result = await generate_texture(
                prompt=job_input.get("prompt", ""),
                seed=job_input.get("seed"),
                job_id=job_id,
                blob_token=blob_token,
                on_progress=on_progress,
            )
        elif job_type == "rotation":
            result = await generate_rotation(
                input_image_url=job_input.get("input_image_url", ""),
                elevation=job_input.get("elevation", 20),
                job_id=job_id,
                blob_token=blob_token,
                on_progress=on_progress,
            )
        else:
            return {"status": "failed", "error": f"Unknown job type: {job_type}"}

        return result
    except Exception as e:
        error_msg = str(e)
        print(f"Job {job_id} failed: {error_msg}")
        return {"status": "failed", "error": error_msg}


if __name__ == "__main__":
    print("Starting GenSprite RunPod worker...")
    runpod.serverless.start({
        "handler": handler,
    })
