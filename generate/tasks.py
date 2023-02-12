"""Celery tasks for generating images."""
import os
import sys
from django.conf import settings

from celery import shared_task

if 'celery' in sys.argv:
    from diffusers import StableDiffusionPipeline

    import torch
else:
    # Avoid loading these on the web server.
    StableDiffusionPipeline = None
    torch = None


@shared_task()
def generate_image(prompt: str) -> None:
    """Generate an image."""
    # model_path = os.path.join(
    #    settings.MODEL_DIRS["stable-diffusion"],
    #    settings.DEFAULT_MODEL_CONFIG["stable-diffusion"],
    # )
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    image = pipe(prompt=prompt).images[0]
    image.save("image.png")
