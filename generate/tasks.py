"""Celery tasks for generating images."""
import logging
import os
import sys
from random import randint
from types import ModuleType
from typing import Optional, Tuple

from django.conf import settings
from django.utils import timezone

from celery import current_app, shared_task

from generate.models import GeneratedImage, RenderWorkerDevice

if "HardDiffusion" in sys.argv:
    import diffusers
    import torch

    def run_safety_checker(self, image, device, dtype):
        """Disabled."""
        has_nsfw_concept = None
        return image, has_nsfw_concept

    diffusers.StableDiffusionPipeline.run_safety_checker = run_safety_checker
    from diffusers import StableDiffusionPipeline
else:
    # Avoid loading these on the web server.
    StableDiffusionPipeline: StableDiffusionPipeline = None  # type: ignore
    torch: ModuleType = None  # type: ignore
HOSTNAME = settings.HOSTNAME
logger = logging.getLogger("HardDiffusion")


@shared_task(bind=True, queue="render", max_concurrency=1)
def generate_image(
    self,
    prompt: str = "An astronaut riding a horse on the moon.",
    model_path_or_name: str = "CompVis/stable-diffusion-v1-4",
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512,
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

    generated_image = GeneratedImage.objects.filter(task_id=task_id).first()
    if generated_image is None:
        raise ValueError(f"GeneratedImage for Task {task_id} not found")
    start = timezone.now()
    pipe = StableDiffusionPipeline.from_pretrained(model_path_or_name)
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda")
    if seed:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(randint(0, 2**32))
    image = pipe(generator=generator, **params).images[0]
    seed = generator.initial_seed()
    end = timezone.now()
    duration = (end - start).total_seconds()
    generated_image.seed = seed
    generated_image.duration = duration
    generated_image.generated_at = end
    filename = generated_image.filename
    generated_image.host = HOSTNAME
    image.save(os.path.join(settings.MEDIA_ROOT, filename))
    generated_image.save()
    return filename, HOSTNAME


@shared_task(bind=True, queue="render")
def health_check(self, **kwargs):
    device_count = torch.cuda.device_count()
    render_devices = RenderWorkerDevice.objects.filter(host=HOSTNAME).all()
    render_devices_by_id = {}
    new = False
    now = timezone.now()
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

    for device in range(device_count):
        render_device = render_devices_by_id[device]
        device_free, device_memory = torch.cuda.mem_get_info(device)
        device_cached = torch.cuda.memory_cached(device)
        render_device.total_memory = device_memory
        render_device.allocated_memory = device_memory - device_free
        render_device.cached_memory = device_cached
        render_device.last_update_at = now
        if new:
            render_device.save()

    if not new:
        RenderWorkerDevice.objects.bulk_update(
            render_devices,
            ["total_memory", "allocated_memory", "cached_memory", "last_update_at"],
        )
    return "Saul Goodman"
