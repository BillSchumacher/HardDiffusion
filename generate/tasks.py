"""Celery tasks for generating images."""
from typing import Optional, Tuple

from django.conf import settings
from django.utils import timezone

from celery import shared_task

import torch

from generate.image import render_image, save_generated_image
from generate.models import GeneratedImage, RenderWorkerDevice
from generate.render_devices import get_render_devices_by_id, update_render_device_stats


HOSTNAME = settings.HOSTNAME


def ensure_model_name(model_path_or_name) -> str:
    """Ensure the model name is set."""
    return model_path_or_name or settings.DEFAULT_TEXT_TO_IMAGE_MODEL


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
    start = timezone.now()
    image, seed = render_image(_model_path_or_name, nsfw, seed, params, generated_image)
    end = timezone.now()
    duration = (end - start).total_seconds()
    save_generated_image(generated_image, image, seed, duration, end)
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
