"""Views for the generate app."""
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from celery import Task
from pydantic import Json

from generate.models import GeneratedImage
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
        num_inference_steps = int(request.POST.get("num_inference_steps", 50))
        height = int(request.POST.get("height", 512))
        width = int(request.POST.get("width", 512))
        seed = request.POST.get("seed", None)
        params = {
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
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


def images(request) -> JsonResponse:
    """Show generated images."""
    generated_images = GeneratedImage.objects.all().order_by("-id")[:10]
    return JsonResponse(
        {
            "images": [
                {
                    "task_id": image.filename[:-4],
                    "filename": image.filename,
                    "host": image.host,
                    "created_at": image.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "generated_at": image.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ") if image.generated_at else None,
                    "prompt": image.prompt,
                    "duration": f"{image.duration:.2f} seconds" if image.duration else None,
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
