from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.


def index(request) -> HttpResponse:
    """Generate an image."""
    return render(request, "train.html")
