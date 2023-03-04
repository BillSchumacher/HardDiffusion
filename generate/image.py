"""Image utilities."""

import base64
import json
import os
from datetime import datetime
from io import BytesIO
from random import randint
from typing import Any, Callable, Dict, List, Optional, Tuple

from django.conf import settings

import numpy as np
import torch
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from diffusers import EulerDiscreteScheduler
from PIL import Image

from generate.models import GeneratedImage
from generate.noise import decode_latents
from generate.pipeline import get_pipeline

HOSTNAME = settings.HOSTNAME
CHANNEL_LAYER = get_channel_layer()
GROUP_SEND = async_to_sync(CHANNEL_LAYER.group_send)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Get the generator for the given seed.

    Args:
        seed (int): The seed to use.

    Returns:
        torch.Generator: The generator with specified seed or a random seed.
    """
    generator = torch.Generator("cuda")
    return generator.manual_seed(seed or randint(0, 2**32))


def render_image(
    model_path_or_name: str,
    nsfw: bool,
    seed: Optional[int],
    params: dict[str, Any],
    callback: Callable,
    callback_steps: int,
    callback_args: List[Any],
    callback_kwargs: Dict[str, Any],
) -> Tuple[Image.Image, int]:
    """Render the image.

    Args:
        model_path_or_name (str): The path or name of the model to use.
        nsfw (bool): Whether to use the NSFW model.
        seed (int): The seed to use.
        params (dict): The parameters to use.
        callback (callable): The callback to use.
        callback_steps (int): The number of steps between callbacks.
        callback_args (list): The arguments to pass to the callback.
        callback_kwargs (dict): The keyword arguments to pass to the callback.

    Returns:
        tuple: The generated image and the seed used to generate it.
    """
    try:
        pipe = get_pipeline(model_path_or_name, nsfw)
    except OSError as ex:
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
    image = merged_pipe(
        generator=generator,
        callback=callback,
        callback_steps=callback_steps,
        callback_args=callback_args,
        callback_kwargs=callback_kwargs,
        **params,
    ).images[0]
    seed = generator.initial_seed()
    return image, seed


def save_error(generated_image: GeneratedImage) -> None:
    """Save the error to the database.

    Args:
        generated_image (GeneratedImage): The generated image to save.

    Returns:
        None
    """
    generated_image.host = HOSTNAME
    generated_image.error = True
    generated_image.save()
    GROUP_SEND(
        "generate",
        {
            "type": "event_message",
            "event": "image_errored",
            "message": str(generated_image.task_id),
        },
    )


def save_generated_image(
    generated_image: GeneratedImage,
    image: Image.Image,
    seed: int,
    duration: float,
    end: datetime,
) -> None:
    """Save the generated image to the database and filesystem.

    Args:
        generated_image (GeneratedImage): The generated image to save.
        image (torch.Tensor): The image to save.
        seed (int): The seed used to generate the image.
        duration (float): The duration of the generation.
        end (datetime.datetime): The end time of the generation.

    Returns:
        None
    """
    generated_image.seed = seed
    generated_image.duration = duration
    generated_image.generated_at = end
    generated_image.host = HOSTNAME
    image.save(os.path.join(settings.MEDIA_ROOT, generated_image.filename))
    generated_image.save()
    GROUP_SEND(
        "generate",
        {
            "type": "event_message",
            "event": "image_generated",
            "message": str(generated_image.task_id),
        },
    )


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (np.ndarray): The image or batch of images to convert.

    Returns:
        PIL.Image.Image: The converted image or batch of images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return (
        [Image.fromarray(image.squeeze(), mode="L") for image in images]
        if images.shape[-1] == 1
        else [Image.fromarray(image) for image in images]
    )


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
        img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        img = None

    event = {
        "task_id": task_id,
        "step": step + 1,
        "total_steps": total_steps,
        "message": message,
        "image": img,
    }
    await CHANNEL_LAYER.group_send(
        "generate",
        {
            "type": "event_message",
            "event": "image_generating",
            "message": json.dumps(event),
        },
    )


GENERATE_IMAGE_STATUS_CALLBACK = async_to_sync(generate_image_status)
