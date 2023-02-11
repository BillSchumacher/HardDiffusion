"""Views for the generate app."""
from django.http import HttpResponse
from django.shortcuts import render

from celery import Task

from generate.tasks import generate_image


def index(request) -> HttpResponse:
    """Generate an image."""
    if request.method == "POST":
        _generate_image: Task = generate_image
        _generate_image.delay(request.POST["prompt"])
    return render(request, "generate.html")
