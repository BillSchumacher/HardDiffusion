"""Image utilities."""

import os
from random import randint

from django.conf import settings

import torch
from diffusers import EulerDiscreteScheduler

from generate.pipeline import get_pipeline

HOSTNAME = settings.HOSTNAME


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
    pipe = pipe.to("cuda")
    generator = get_generator(seed)
    if isinstance(model_path_or_name, list):
        merged_pipe = pipe.merge(
            model_path_or_name[1:],
            interp="sigmoid",
            alpha=0.4,
        )
    else:
        merged_pipe = pipe

    merged_pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    image = merged_pipe(generator=generator, **params).images[0]
    seed = generator.initial_seed()
    return image, seed


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
