"""Celery tasks for generating images."""
import os
import sys
from types import ModuleType
from typing import Optional, Tuple

from django.conf import settings
from django.utils import timezone

from celery import shared_task

from generate.models import GeneratedImage

if "HardDiffusion" in sys.argv:
    import torch
    import diffusers
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


@shared_task(bind=True)
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
