"""
Main RunPod handler for GenSprite worker.
Routes jobs to appropriate pipelines and handles webhooks.
"""

import os
import asyncio
import runpod
from typing import Any

# Import pipelines
from pipelines.sprite import generate_sprite
from pipelines.texture import generate_texture
from pipelines.rotation import generate_rotation

# Import utilities
from utils.webhook import send_completed, send_failed, send_progress


# Environment variables
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "")


async def process_job(job: dict) -> dict:
    """
    Process a job request from RunPod.

    Expected job input format:
    {
        "job_id": "abc123",           # ID in main app's database
        "job_type": "sprite",         # sprite | texture | rotation
        "webhook_url": "https://...", # Optional override
        "webhook_secret": "...",      # Optional override
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
    webhook_url = job_input.get("webhook_url", WEBHOOK_URL)
    webhook_secret = job_input.get("webhook_secret", WEBHOOK_SECRET)
    blob_token = job_input.get("blob_token", BLOB_TOKEN)

    if not job_id:
        return {"error": "Missing job_id"}
    if not job_type:
        return {"error": "Missing job_type"}
    if not blob_token:
        return {"error": "Missing blob_token"}

    # Progress callback for updates
    async def on_progress(progress: int, stage: str):
        if webhook_url and webhook_secret:
            await send_progress(
                job_id=job_id,
                job_type=job_type,
                progress=progress,
                current_stage=stage,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
            )

    try:
        result = None

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
            error_msg = f"Unknown job type: {job_type}"
            if webhook_url and webhook_secret:
                await send_failed(
                    job_id=job_id,
                    job_type=job_type,
                    error=error_msg,
                    webhook_url=webhook_url,
                    webhook_secret=webhook_secret,
                )
            return {"error": error_msg}

        # Send completion webhook
        if webhook_url and webhook_secret:
            await send_completed(
                job_id=job_id,
                job_type=job_type,
                result=result,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
            )

        return {"status": "completed", "result": result}

    except Exception as e:
        error_msg = str(e)
        print(f"Job {job_id} failed: {error_msg}")

        # Send failure webhook
        if webhook_url and webhook_secret:
            await send_failed(
                job_id=job_id,
                job_type=job_type,
                error=error_msg,
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
            )

        return {"error": error_msg}


def handler(job: dict) -> dict:
    """
    Synchronous handler wrapper for RunPod.
    """
    return asyncio.get_event_loop().run_until_complete(process_job(job))


if __name__ == "__main__":
    print("Starting GenSprite RunPod worker...")
    runpod.serverless.start({"handler": handler})
