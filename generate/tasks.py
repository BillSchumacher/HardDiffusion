"""Celery tasks for generating images."""
import os
import sys
from typing import Tuple
from uuid import uuid4

from django.conf import settings

from celery import shared_task

from generate.models import GeneratedImage

if "HardDiffusion" in sys.argv:
    from diffusers import StableDiffusionPipeline

    # import torch
else:
    # Avoid loading these on the web server.
    StableDiffusionPipeline: StableDiffusionPipeline = None  # type: ignore
    # torch = None
HOSTNAME = settings.HOSTNAME


@shared_task()
def generate_image(
    prompt: str, model_path_or_name: str = "CompVis/stable-diffusion-v1-4"
) -> Tuple[str, str]:
    """Generate an image."""
    # model_path = os.path.join(
    #    settings.MODEL_DIRS["stable-diffusion"],
    #    settings.DEFAULT_MODEL_CONFIG["stable-diffusion"],
    # )
    pipe = StableDiffusionPipeline.from_pretrained(model_path_or_name)
    # pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    image = pipe(prompt=prompt).images[0]
    filename = f"{uuid4().hex}.png"
    image.save(os.path.join(settings.MEDIA_ROOT, filename))

    GeneratedImage(filename=filename, host=HOSTNAME).save()
    return filename, HOSTNAME
