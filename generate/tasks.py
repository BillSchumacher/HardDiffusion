"""Celery tasks for generating images."""
from typing import Optional, Tuple

from django.conf import settings
from django.utils import timezone

import torch
from celery import shared_task

from generate.image import (
    GENERATE_IMAGE_STATUS_CALLBACK,
    render_image,
    save_error,
    save_generated_image,
)
from generate.models import GeneratedImage, RenderWorkerDevice
from generate.render_devices import get_render_devices_by_id, update_render_device_stats
from HardDiffusion.logs import logger

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
    """Generate an image.
    
    Args:
        prompt (str, optional): The prompt to use. 
            Defaults to "An astronaut riding a horse on the moon.".
        negative_prompt (Optional[str], optional): 
            The negative prompt to use. Defaults to None.
        model_path_or_name (Optional[str], optional): The model path or name. 
            Defaults to None.
        seed (Optional[int], optional): The seed to use. Defaults to None.
        guidance_scale (float, optional): The guidance scale to use. Defaults to 7.5.
        num_inference_steps (int, optional): The number of inference steps to use.
            Defaults to 50.
        height (int, optional): The height of the image. Defaults to 512.
        width (int, optional): The width of the image. Defaults to 512.
        nsfw (bool, optional): Whether to use the NSFW model. Defaults to False.
        callback_steps (int, optional): The number of steps between callbacks. 
            Defaults to 1.
        preview_image (bool, optional): Whether to preview the image. Defaults to False.

    Returns:
        Tuple[str, str]: The image filename and seed.
    """
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
    try:
        start = timezone.now()
        image, seed = render_image(
            _model_path_or_name,
            nsfw,
            seed,
            params,
            callback=GENERATE_IMAGE_STATUS_CALLBACK,
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
    except Exception as e:
        logger.error("%s", e)
        save_error(generated_image)
        raise e
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
