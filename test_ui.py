"""
Gradio UI for testing rotation refinement pipeline.
Run: python test_ui.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/workspace/models")

import gradio as gr
from PIL import Image
import numpy as np


def test_refine_only(image):
    """Test just the refinement step."""
    from pipelines.rotation import refine_frame

    if image is None:
        return None

    img = Image.fromarray(image)
    refined = refine_frame(img, prompt="game sprite, high quality, detailed, sharp")
    return refined


def test_full_rotation(image, elevation):
    """Test full rotation pipeline (without upload)."""
    import asyncio
    from pipelines.rotation import (
        preprocess_image,
        get_sv3d_pipeline,
        refine_frame,
        DIRECTION_INDICES,
    )
    from models.loader import get_rembg_session
    from rembg import remove
    import torch
    import math

    if image is None:
        return [None] * 8

    img = Image.fromarray(image)

    # Preprocess
    processed = preprocess_image(img, size=576)

    # Run SV3D
    pipe = get_sv3d_pipeline()
    num_frames = 21
    polar_rad = [math.radians(90 - elevation)] * num_frames
    azimuths_rad = [math.radians(i * 360 / num_frames) for i in range(num_frames)]

    generator = torch.Generator(device="cuda").manual_seed(42)
    output = pipe(
        processed,
        height=576,
        width=576,
        num_frames=num_frames,
        polars_rad=polar_rad,
        azimuths_rad=azimuths_rad,
        decode_chunk_size=8,
        generator=generator,
        num_inference_steps=20,
    )
    frames = output.frames[0]

    # Process 8 directions
    rembg_session = get_rembg_session()
    results = []

    for direction in ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]:
        frame_index = DIRECTION_INDICES[direction]
        frame = frames[frame_index]

        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        # Refine
        refined = refine_frame(frame, prompt="game sprite, high quality, detailed, sharp")

        # Remove background
        transparent = remove(
            refined,
            session=rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
        )

        results.append(transparent)

    return results


# Build UI
with gr.Blocks(title="Rotation Pipeline Test") as demo:
    gr.Markdown("# Rotation Pipeline Test")

    with gr.Tab("Test Refinement Only"):
        gr.Markdown("Upload an image to test RealESRGAN + ControlNet Tile refinement")
        with gr.Row():
            refine_input = gr.Image(label="Input (576px SV3D-like)")
            refine_output = gr.Image(label="Refined (1024px)")
        refine_btn = gr.Button("Refine", variant="primary")
        refine_btn.click(test_refine_only, inputs=refine_input, outputs=refine_output)

    with gr.Tab("Test Full Rotation"):
        gr.Markdown("Upload a sprite to test the full SV3D → Refine → Remove BG pipeline")
        with gr.Row():
            rotation_input = gr.Image(label="Input Sprite")
            elevation_slider = gr.Slider(
                minimum=-30, maximum=60, value=20, step=5,
                label="Elevation Angle"
            )
        rotation_btn = gr.Button("Generate 8 Directions", variant="primary")

        with gr.Row():
            outputs = [gr.Image(label=d) for d in ["S", "SW", "W", "NW"]]
        with gr.Row():
            outputs += [gr.Image(label=d) for d in ["N", "NE", "E", "SE"]]

        rotation_btn.click(
            test_full_rotation,
            inputs=[rotation_input, elevation_slider],
            outputs=outputs
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
