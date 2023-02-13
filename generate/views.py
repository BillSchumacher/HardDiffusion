"""Views for the generate app."""
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from celery import Task

from generate.models import GeneratedImage
from generate.tasks import generate_image


def index(request) -> HttpResponse:
    """Generate an image."""
    if request.method == "POST":
        _generate_image: Task = generate_image  # type: ignore
        _generate_image.delay(request.POST["prompt"])
    context = (
        {"use_localhost": "true"}
        if settings.USE_LOCALHOST
        else {"use_localhost": "false"}
    )
    return render(request, "generate.html", context)


def images(request) -> JsonResponse:
    """Show generated images."""
    generated_images = GeneratedImage.objects.all().order_by("-id")[:10]
    return JsonResponse(
        {
            "images": [
                {
                    "filename": image.filename,
                    "host": image.host,
                    "created_at": image.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                for image in generated_images
            ]
        }
    )
