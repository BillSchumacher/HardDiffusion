"""Views for the generate app."""
import os

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.utils import save_images, log


def index(request) -> HttpResponse:
    """Generate an image."""
    if request.method == 'POST':
        context = sdkit.Context()

        # set the path to the model file on the disk 
        # (.ckpt or .safetensors file)
        context.model_paths['stable-diffusion'] = os.path.join(
            settings.MODEL_DIRS['stable-diffusion'],
            settings.DEFAULT_MODEL['stable-diffusion']
        )
        load_model(context, 'stable-diffusion')

        # generate the image
        images = generate_images(
            context,
            prompt=request.POST['prompt'],
            seed=42,
            width=512,
            height=512
        )

        # save the image
        save_images(images, dir_path=settings.BASE_DIR)

        log.info("Generated images!")
    return render(request, 'index.html')
