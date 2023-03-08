"""Generate tasks."""
from datetime import datetime
from typing import Optional, Tuple

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

from generate.models import GeneratedImage, RenderWorkerDevice

CHANNEL_LAYER = get_channel_layer()
GROUP_SEND = async_to_sync(CHANNEL_LAYER.group_send)


@shared_task(name="generate_image_status", queue="image_progress")
def generate_image_status_task(event):
    GROUP_SEND(
        "generate",
        {
            "type": "event_message",
            "event": "image_generating",
            "message": event,
        },
    )


@shared_task(name="generate_image_completed", queue="image_progress")
def generate_image_completed_task(image_id, task_id, hostname, start, end, seed):
    generated_image = GeneratedImage.objects.get(pk=image_id)
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)
    duration = (end - start).total_seconds()
    save_generated_image(generated_image, task_id, seed, duration, start, end, hostname)


def save_generated_image(
    generated_image: GeneratedImage,
    task_id: str,
    seed: int,
    duration: float,
    start: datetime,
    end: datetime,
    hostname: str,
) -> None:
    """Save the generated image to the database.

    Args:
        generated_image (GeneratedImage): The generated image to save.
        task_id (str): The task that generated the image.
        image (torch.Tensor): The image to save.
        seed (int): The seed used to generate the image.
        duration (float): The duration of the generation.
        end (datetime.datetime): The end time of the generation.
        hostname (str): The host that the image was saved to.
    Returns:
        None
    """
    generated_image.task_id = task_id
    generated_image.seed = seed
    generated_image.duration = duration
    generated_image.generated_at = end
    generated_image.host = hostname
    generated_image.save()
    GROUP_SEND(
        "generate",
        {
            "type": "event_message",
            "event": "image_generated",
            "message": task_id,
        },
    )


@shared_task(name="generate_image_error", queue="image_progress")
def generate_image_error_task(image_id, task_id, hostname):
    generated_image = GeneratedImage.objects.get(pk=image_id)
    save_error(generated_image, task_id, hostname)


def save_error(generated_image: GeneratedImage, task_id, hostname) -> None:
    """Save the error to the database.

    Args:
        generated_image (GeneratedImage): The generated image to save.

    Returns:
        None
    """
    generated_image.task_id = task_id
    generated_image.host = hostname
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


# @shared_task(queue="render")
def health_check(**kwargs):
    """Check the health of the render worker. Disabled for now"""
    """
    device_count = torch.cuda.device_count()
    now = datetime.now()
    render_devices, render_devices_by_id, new = get_render_devices_by_id(device_count)
    for device_id in range(device_count):
        render_device = render_devices_by_id[device_id]
        update_render_device_stats(device_id, render_device, now, new)
    if not new:
        RenderWorkerDevice.objects.bulk_update(
            render_devices,
            ["total_memory", "allocated_memory", "cached_memory", "last_update_at"],
        )
    """


def get_render_devices_by_id(device_count):
    """Get the existing render workers. Disabled for now"""
    """render_devices = RenderWorkerDevice.objects.filter(host=hostname).all()
    render_devices_by_id = {}
    new = False
    if not render_devices:
        new = True
        for device in range(device_count):
            device_name = torch.cuda.get_device_name(device)
            render_devices_by_id[device] = RenderWorkerDevice(
                device_id=device, host=hostname, device_name=device_name
            )
    else:
        for device in render_devices:
            render_devices_by_id[device.device_id] = device
    return render_devices, render_devices_by_id, new
    """


def update_render_device_stats(device_id, render_device, now, new):
    """Update the render device stats. Disabled for now"""
    """
    device_free, device_memory = torch.cuda.mem_get_info(device_id)
    device_cached = torch.cuda.memory_cached(device_id)
    render_device.total_memory = device_memory
    render_device.allocated_memory = device_memory - device_free
    render_device.cached_memory = device_cached
    render_device.last_update_at = now
    if new:
        render_device.save()
    """


@shared_task(bind=True, name="generate_image", queue="render", max_concurrency=1)
def generate_image(
    self,
    image_id: int,
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
    raise NotImplementedError("This should not run here...")
