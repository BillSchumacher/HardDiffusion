"""Noise functions"""

import torch


def denoise(scheduler, unet, latents, i, t, prompt_embeds, do_classifier_free_guidance,
            guidance_scale, extra_step_kwargs, timesteps, num_warmup_steps,
            progress_bar, callback, callback_steps):
    """denoise the image"""

    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = scheduler.scale_model_input(
        latent_model_input, t
    )

    # predict the noise residual
    noise_pred = unet(
        latent_model_input, t, encoder_hidden_states=prompt_embeds
    ).sample

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(
        noise_pred, t, latents, **extra_step_kwargs
    ).prev_sample

    # call the callback, if provided
    if i == len(timesteps) - 1 or (
        (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
    ):
        progress_bar.update()
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)
    return latents
