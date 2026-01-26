"""
SV3D Pipeline implementation.
Based on: https://github.com/chenguolin/sv3d-diffusers
"""

from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import torch

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.models import UNetSpatioTemporalConditionModel
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipeline,
    StableVideoDiffusionPipelineOutput,
    _append_dims,
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


class StableVideo3DDiffusionPipeline(StableVideoDiffusionPipeline):
    """
    Pipeline for SV3D (Stable Video 3D) multi-view generation.

    Extends StableVideoDiffusionPipeline to support camera angle conditioning
    (polar and azimuth angles) for 3D-consistent multi-view synthesis.
    """

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
        )

    def _get_add_time_ids(
        self,
        noise_aug_strength: float,
        polars_rad: List[float],
        azimuths_rad: List[float],
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        """
        Create the time embeddings for SV3D, including camera angles.
        """
        cond_aug = torch.tensor(
            [noise_aug_strength] * len(polars_rad), dtype=dtype
        ).repeat(batch_size * num_videos_per_prompt, 1)
        polars_rad = torch.tensor(polars_rad, dtype=dtype).repeat(
            batch_size * num_videos_per_prompt, 1
        )
        azimuths_rad = torch.tensor(azimuths_rad, dtype=dtype).repeat(
            batch_size * num_videos_per_prompt, 1
        )

        if do_classifier_free_guidance:
            cond_aug = torch.cat([cond_aug, cond_aug])
            polars_rad = torch.cat([polars_rad, polars_rad])
            azimuths_rad = torch.cat([azimuths_rad, azimuths_rad])

        add_time_ids = [cond_aug, polars_rad, azimuths_rad]

        return add_time_ids

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        polars_rad: List[float],
        azimuths_rad: List[float],
        triangle_cfg_scaling: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 2.5,
        noise_aug_strength: float = 1e-5,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        """
        Generate multi-view images from a single input image using SV3D.

        Args:
            image: Input image (PIL.Image, list of images, or tensor)
            polars_rad: List of polar angles in radians for each frame
            azimuths_rad: List of azimuth angles in radians for each frame
            triangle_cfg_scaling: Use triangle CFG scaling (recommended for SV3D)
            height: Output height (defaults to model's sample_size * vae_scale_factor)
            width: Output width (defaults to model's sample_size * vae_scale_factor)
            num_frames: Number of frames to generate (defaults to model's num_frames)
            num_inference_steps: Number of denoising steps
            min_guidance_scale: Minimum guidance scale for CFG
            max_guidance_scale: Maximum guidance scale for CFG
            noise_aug_strength: Noise augmentation strength
            decode_chunk_size: Chunk size for VAE decoding
            num_videos_per_prompt: Number of videos to generate per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            output_type: Output type ("pil", "pt", "np", "latent")
            return_dict: Whether to return a dict or tuple

        Returns:
            StableVideoDiffusionPipelineOutput with generated frames
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(
            image, device, num_videos_per_prompt, self.do_classifier_free_guidance
        )

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame
        # image_latents [batch, channels, height, width] -> [batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs (with camera angles for SV3D)
        added_time_ids = self._get_add_time_ids(
            noise_aug_strength,
            polars_rad,
            azimuths_rad,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = [a.to(device) for a in added_time_ids]

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, sigmas
        )

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare guidance scale
        if triangle_cfg_scaling:
            # Triangle CFG scaling; the last view is input condition
            guidance_scale = torch.cat([
                torch.linspace(min_guidance_scale, max_guidance_scale, num_frames // 2 + 1)[1:].unsqueeze(0),
                torch.linspace(max_guidance_scale, min_guidance_scale, num_frames - num_frames // 2 + 1)[1:].unsqueeze(0)
            ], dim=-1)
        else:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
