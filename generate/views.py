"""Views for the generate app."""
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
    return render(request, "generate.html")


def images(request) -> JsonResponse:
    """Show generated images."""
    generated_images = GeneratedImage.objects.all()
    return JsonResponse(
        {
            "images": [
                {"filename": image.filename, "host": image.host}
                for image in generated_images
            ]
        }
    )
