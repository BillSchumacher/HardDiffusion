"""Noise functions"""

from typing import Any

import numpy as np
import torch


def denoise(
    scheduler,
    unet,
    latents,
    step,
    current_timestep,
    prompt_embeds,
    do_classifier_free_guidance,
    guidance_scale,
    extra_step_kwargs,
    timesteps,
    num_warmup_steps,
    progress_bar,
    callback,
    callback_steps,
    callback_args,
    callback_kwargs,
    vae=None,
):
    """denoise the image

    Args:
        scheduler (Scheduler): scheduler
        unet (UNet): unet
        latents (torch.Tensor): latents
        step (int): step
        current_timestep (torch.Tensor): current timestep
        prompt_embeds (torch.Tensor): prompt embeds
        do_classifier_free_guidance (bool): do classifier free guidance
        guidance_scale (float): guidance scale
        extra_step_kwargs (dict): extra step kwargs
        timesteps (list): timesteps
        num_warmup_steps (int): num warmup steps
        progress_bar (tqdm): progress bar
        callback (function): callback
        callback_steps (int): callback steps
        callback_args (list): callback args
        callback_kwargs (dict): callback kwargs
        vae (VAE): VAE model

    Returns:
        latents (torch.Tensor): latents
    """

    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = scheduler.scale_model_input(
        latent_model_input, current_timestep
    )

    # predict the noise residual
    noise_pred = unet(
        latent_model_input, current_timestep, encoder_hidden_states=prompt_embeds
    ).sample

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(
        noise_pred, current_timestep, latents, **extra_step_kwargs
    ).prev_sample

    # call the callback, if provided
    if step == len(timesteps) - 1 or (
        (step + 1) > num_warmup_steps and (step + 1) % scheduler.order == 0
    ):
        progress_bar.update()
        if callback is not None and step % callback_steps == 0:
            callback(
                step,
                current_timestep,
                latents,
                *callback_args,
                vae=vae,
                **callback_kwargs,
            )
    return latents


@torch.no_grad()
def decode_latents(vae: Any, latents: torch.Tensor) -> np.ndarray:
    """decode latents to image

    Args:
        vae (VAE): VAE model
        latents (torch.Tensor): latents

        Returns:
            image (np.ndarray): image
    """
    _latents: torch.Tensor = 1 / 0.18215 * latents
    image = vae.decode(_latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead
    #  and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
