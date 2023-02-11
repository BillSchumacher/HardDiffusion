"""Celery tasks for generating images."""
import os

from django.conf import settings

import sdkit
from celery import shared_task
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log, save_images


@shared_task()
def generate_image(prompt: str) -> None:
    """Generate an image."""
    context = sdkit.Context()

    # set the path to the model file on the disk
    # (.ckpt or .safetensors file)
    context.model_paths["stable-diffusion"] = os.path.join(
        settings.MODEL_DIRS["stable-diffusion"],
        settings.DEFAULT_MODEL["stable-diffusion"],
    )

    context.model_configs["stable-diffusion"] = os.path.join(
        settings.MODEL_DIRS["stable-diffusion"],
        settings.DEFAULT_MODEL_CONFIG["stable-diffusion"],
    )
    load_model(context, "stable-diffusion")

    # generate the image
    images = generate_images(context, prompt=prompt, seed=42, width=512, height=512)

    # save the image
    save_images(images, dir_path=settings.BASE_DIR)

    log.info("Generated images!")
