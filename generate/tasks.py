"""Celery tasks for generating images."""
import logging
import os
import sys
from random import randint
from types import ModuleType
from typing import Callable, Optional, Tuple

from django.conf import settings
from django.utils import timezone

from celery import shared_task

from generate.models import GeneratedImage, RenderWorkerDevice

if "HardDiffusion" in sys.argv or "test" in sys.argv:
    import diffusers
    import torch

    def run_safety_checker(_, image, __, ___):
        """Disabled."""
        has_nsfw_concept = None
        return image, has_nsfw_concept

    original_run_safety_checker = diffusers.StableDiffusionPipeline.run_safety_checker
    diffusers.StableDiffusionPipeline.run_safety_checker = run_safety_checker
    from diffusers import StableDiffusionPipeline
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        load_pipeline_from_original_stable_diffusion_ckpt,
    )
else:
    # Avoid loading these on the web server.
    StableDiffusionPipeline: StableDiffusionPipeline = None  # type: ignore
    torch: ModuleType = None  # type: ignore
    load_pipeline_from_original_stable_diffusion_ckpt: Callable = None  # type: ignore


HOSTNAME = settings.HOSTNAME
logger = logging.getLogger("HardDiffusion")


def get_pipeline(model_path_or_name, nsfw):
    """Get the pipeline for the given model path or name."""
    if model_path_or_name.startswith("./") and model_path_or_name.endswith(".ckpt"):
        return load_pipeline_from_original_stable_diffusion_ckpt(
            model_path_or_name,
            model_path_or_name.replace(".ckpt", ".yaml"),
        )
    StableDiffusionPipeline.run_safety_checker = (
        run_safety_checker if nsfw else original_run_safety_checker
    )
    return StableDiffusionPipeline.from_pretrained(model_path_or_name)


def ensure_model_name(model_path_or_name) -> str:
    """Ensure the model name is set."""
    return model_path_or_name or settings.DEFAULT_TEXT_TO_IMAGE_MODEL


def save_error(generated_image):
    """Save the error to the database."""
    generated_image.host = HOSTNAME
    generated_image.error = True
    generated_image.save()


def save_generated_image(generated_image, image, seed, duration, end):
    """Save the generated image to the database and filesystem."""
    generated_image.seed = seed
    generated_image.duration = duration
    generated_image.generated_at = end
    generated_image.host = HOSTNAME
    image.save(os.path.join(settings.MEDIA_ROOT, generated_image.filename))
    generated_image.save()


def get_generator(seed):
    """Get the generator for the given seed."""
    generator = torch.Generator("cuda")
    return generator.manual_seed(seed or randint(0, 2**32))


def render_image(model_path_or_name, nsfw, seed, params, generated_image):
    """Render the image."""
    try:
        pipe = get_pipeline(model_path_or_name, nsfw)
    except OSError as ex:
        print(ex)
        save_error(generated_image)
        raise RuntimeError("Error loading model") from ex
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    generator = get_generator(seed)
    image = pipe(generator=generator, **params).images[0]
    seed = generator.initial_seed()
    return image, seed


@shared_task(bind=True, queue="render", max_concurrency=1)
def generate_image(
    self,
    prompt: str = "An astronaut riding a horse on the moon.",
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


def get_render_devices_by_id(device_count):
    """Get the existing render workers."""
    render_devices = RenderWorkerDevice.objects.filter(host=HOSTNAME).all()
    render_devices_by_id = {}
    new = False
    if not render_devices:
        new = True
        for device in range(device_count):
            device_name = torch.cuda.get_device_name(device)
            render_devices_by_id[device] = RenderWorkerDevice(
                device_id=device, host=HOSTNAME, device_name=device_name
            )
    else:
        for device in render_devices:
            render_devices_by_id[device.device_id] = device
    return render_devices, render_devices_by_id, new


def update_render_device_stats(device_id, render_device, now, new):
    """Update the render device stats."""
    device_free, device_memory = torch.cuda.mem_get_info(device_id)
    device_cached = torch.cuda.memory_cached(device_id)
    render_device.total_memory = device_memory
    render_device.allocated_memory = device_memory - device_free
    render_device.cached_memory = device_cached
    render_device.last_update_at = now
    if new:
        render_device.save()


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
