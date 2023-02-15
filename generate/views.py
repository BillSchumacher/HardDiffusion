"""Views for the generate app."""
from collections import defaultdict
from datetime import timedelta
from typing import Optional

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone

from celery import Task

from generate.models import GeneratedImage, RenderWorkerDevice
from generate.tasks import generate_image


def index(request) -> HttpResponse:
    """Generate an image."""
    context = (
        {"use_localhost": "true"}
        if settings.USE_LOCALHOST
        else {"use_localhost": "false"}
    )
    return render(request, "generate.html", context)


def queue_prompt(request) -> JsonResponse:
    """Queue a prompt to be generated."""
    if request.method == "POST":
        _generate_image: Task = generate_image  # type: ignore
        guidance_scale = float(request.POST.get("guidance_scale", 7.5))
        inference_steps = int(request.POST.get("inference_steps", 50))
        height = int(request.POST.get("height", 512))
        width = int(request.POST.get("width", 512))
        seed = request.POST.get("seed", None)
        if seed:
            seed = int(seed)
        params = {
            "guidance_scale": guidance_scale,
            "num_inference_steps": inference_steps,
            "height": height,
            "width": width,
            "seed": seed,
        }
        result = _generate_image.apply_async(
            kwargs=dict(prompt=request.POST["prompt"], **params), countdown=2
        )
        task_id = result.id
        GeneratedImage(
            prompt=request.POST["prompt"],
            task_id=task_id,
            **params,
        ).save()

        return JsonResponse({"error": None, "task_id": task_id})
    return JsonResponse({"error": "POST request required", "task_id": None})


def images(request, last: Optional[int] = None) -> JsonResponse:
    """Show generated images."""
    generated_images = GeneratedImage.objects.all().order_by("-id")[:10]
    return JsonResponse(
        {
            "images": [
                {   
                    "id": image.id,
                    "task_id": image.filename[:-4],
                    "filename": image.filename,
                    "host": image.host,
                    "created_at": image.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "generated_at": image.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if image.generated_at
                    else None,
                    "prompt": image.prompt,
                    "duration": f"{image.duration:.2f} seconds"
                    if image.duration
                    else None,
                    "seed": image.seed,
                    "guidance_scale": image.guidance_scale,
                    "num_inference_steps": image.num_inference_steps,
                    "height": image.height,
                    "width": image.width,
                }
                for image in generated_images
            ]
        }
    )


def renderer_health(request) -> HttpResponse:
    """Check GPU memory"""
    render_devices = RenderWorkerDevice.objects.filter(
        last_update_at__gt=timezone.now() - timedelta(seconds=30)
    )
    div = 1024**3
    results_by_hostname = defaultdict(list)
    for device in render_devices:
        results_by_hostname[device.host].append(
            {
                "id": device.device_id,
                "name": device.device_name,
                "usage": f"Memory Usage: {round((device.allocated_memory-device.cached_memory)/div,1)} GB",
                "memory": f"Total: {round(device.total_memory/div,1)} GB",
                "device_allocated": f"Allocated: {round(device.allocated_memory/div,1)} GB",
                "device_cached": f"Cached: {round(device.cached_memory/div,1)} GB",
            }
        )
    return JsonResponse({"results": results_by_hostname})
