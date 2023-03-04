"""Celery tasks for generating images."""
import base64
import json
from io import BytesIO
from pprint import pprint
from typing import Optional, Tuple

from django.conf import settings
from django.utils import timezone

import torch
from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

from generate.image import numpy_to_pil, render_image, save_generated_image
from generate.models import GeneratedImage, RenderWorkerDevice
from generate.noise import decode_latents
from generate.render_devices import get_render_devices_by_id, update_render_device_stats

HOSTNAME = settings.HOSTNAME


def ensure_model_name(model_path_or_name: str) -> str:
    """Ensure the model name is set.

    If the model name is not set, use the default model name.

    Args:
        model_path_or_name (str): The model path or name.

    Returns:
        str: The model name.
    """
    return model_path_or_name or settings.DEFAULT_TEXT_TO_IMAGE_MODEL


async def generate_image_status(
    step: int,
    timestep: Optional[torch.FloatTensor],
    latents: Optional[torch.Tensor],
    task_id,
    total_steps=None,
    vae=None,
    message=None,
    preview_image=False,
    *args,
    **kwargs,
) -> None:
    """Status callback for image generation.

    Preview images are only sent if the VAE is provided.

    Preview images reduce the performance of the generation process, by half of more.
    Args:
        step (int): The current step.
        timestep (torch.FloatTensor): The current timestep, unused currently.
        latents (torch.Tensor): The current latents.
        task_id (str): The task ID.
        message (str): The message to send.
        preview_image (bool): Whether to send a preview image.

    Returns:
        None
    """
    if latents is not None and vae and preview_image:
        image = decode_latents(vae, latents)
    else:
        image = None

    if image is not None and preview_image:
        buffered = BytesIO()
        image = numpy_to_pil(image)
        image[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        img_str = None

    event = {
        "task_id": task_id,
        "step": step + 1,
        "total_steps": total_steps,
        "message": message,
        "image": img_str,
    }
    channel_layer = get_channel_layer()
    await channel_layer.group_send(
        "generate",
        {
            "type": "event_message",
            "event": "image_generating",
            "message": json.dumps(event),
        },
    )


@shared_task(bind=True, queue="render", max_concurrency=1)
def generate_image(
    self,
    prompt: str = "An astronaut riding a horse on the moon.",
    negative_prompt: Optional[str] = None,
    model_path_or_name: Optional[str] = None,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512,
    nsfw: bool = False,
    callback_steps: int = 1,
    preview_image: bool = False,
) -> Tuple[str, str]:
    """Generate an image."""
    task_id = self.request.id
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "height": height,
        "width": width,
    }
    _model_path_or_name = ensure_model_name(model_path_or_name)
    generated_image = GeneratedImage.objects.filter(task_id=task_id).first()
    if generated_image is None:
        raise ValueError(f"GeneratedImage for Task {task_id} not found")
    callback = async_to_sync(generate_image_status)
    start = timezone.now()
    image, seed = render_image(
        _model_path_or_name,
        nsfw,
        seed,
        params,
        generated_image,
        callback=callback,
        callback_steps=callback_steps,
        callback_args=[task_id],
        callback_kwargs={
            "total_steps": num_inference_steps,
            "preview_image": preview_image,
        },
    )
    end = timezone.now()
    duration = (end - start).total_seconds()
    save_generated_image(generated_image, image, seed, duration, end)
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "generate",
        {"type": "event_message", "event": "image_generated", "message": str(task_id)},
    )
    return generated_image.filename, HOSTNAME


@shared_task(queue="render")
def health_check(**kwargs):
    """Check the health of the render worker."""
    device_count = torch.cuda.device_count()
    now = timezone.now()
    render_devices, render_devices_by_id, new = get_render_devices_by_id(device_count)
    for device_id in range(device_count):
        render_device = render_devices_by_id[device_id]
        update_render_device_stats(device_id, render_device, now, new)
    if not new:
        RenderWorkerDevice.objects.bulk_update(
            render_devices,
            ["total_memory", "allocated_memory", "cached_memory", "last_update_at"],
        )
