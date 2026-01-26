"""
Webhook notification utilities for reporting job status back to main app.
"""

import os
import hmac
import hashlib
import json
import httpx
from typing import Optional, Any


async def send_webhook(
    job_id: str,
    job_type: str,
    status: str,
    webhook_url: str,
    webhook_secret: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    progress: Optional[int] = None,
    current_stage: Optional[str] = None,
) -> bool:
    """
    Send a webhook notification to the main app.

    Args:
        job_id: The job ID in the main app's database
        job_type: Type of job (sprite, rotation, texture)
        status: Job status (processing, completed, failed)
        webhook_url: URL to send the webhook to
        webhook_secret: Secret for HMAC signature
        result: Result data (URLs, etc.) for completed jobs
        error: Error message for failed jobs
        progress: Progress percentage (0-100)
        current_stage: Human-readable stage description

    Returns:
        True if webhook was sent successfully
    """
    payload = {
        "job_id": job_id,
        "job_type": job_type,
        "status": status,
    }

    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    if progress is not None:
        payload["progress"] = progress
    if current_stage is not None:
        payload["current_stage"] = current_stage

    # Create HMAC signature
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    signature = hmac.new(
        webhook_secret.encode(),
        payload_json.encode(),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Signature": signature,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                content=payload_json,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"Webhook failed: {e}")
        return False


async def send_progress(
    job_id: str,
    job_type: str,
    progress: int,
    current_stage: str,
    webhook_url: str,
    webhook_secret: str,
) -> bool:
    """
    Send a progress update webhook.

    Args:
        job_id: The job ID
        job_type: Type of job
        progress: Progress percentage (0-100)
        current_stage: Human-readable stage description
        webhook_url: URL to send the webhook to
        webhook_secret: Secret for HMAC signature

    Returns:
        True if webhook was sent successfully
    """
    return await send_webhook(
        job_id=job_id,
        job_type=job_type,
        status="processing",
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        progress=progress,
        current_stage=current_stage,
    )


async def send_completed(
    job_id: str,
    job_type: str,
    result: dict,
    webhook_url: str,
    webhook_secret: str,
) -> bool:
    """
    Send a job completion webhook.

    Args:
        job_id: The job ID
        job_type: Type of job
        result: Result data (URLs, etc.)
        webhook_url: URL to send the webhook to
        webhook_secret: Secret for HMAC signature

    Returns:
        True if webhook was sent successfully
    """
    return await send_webhook(
        job_id=job_id,
        job_type=job_type,
        status="completed",
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        result=result,
        progress=100,
        current_stage="Completed",
    )


async def send_failed(
    job_id: str,
    job_type: str,
    error: str,
    webhook_url: str,
    webhook_secret: str,
) -> bool:
    """
    Send a job failure webhook.

    Args:
        job_id: The job ID
        job_type: Type of job
        error: Error message
        webhook_url: URL to send the webhook to
        webhook_secret: Secret for HMAC signature

    Returns:
        True if webhook was sent successfully
    """
    return await send_webhook(
        job_id=job_id,
        job_type=job_type,
        status="failed",
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        error=error,
    )
