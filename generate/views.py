"""Views for the generate app.

    Contains the following views:
    - index: The main page for the generate app.
    - queue_prompt: Queue a prompt to be generated.
    - csrf_form: CSRF endpoint to get the middleware to set the token.
    - images: Return generated images data as JSON.
    - renderer_health: Check health of all renderer devices.
    - renderer_status: The rendered status page.

    Also contains the following viewsets:
    - GeneratedImageViewSet: Viewset for the GeneratedImage model.
"""
from collections import defaultdict
from datetime import timedelta
from typing import Optional

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from dynamic_rest.viewsets import DynamicModelViewSet

# from rest_framework.decorators import action
from rest_framework import permissions

# from rest_framework.response import Response
from generate.models import GeneratedImage, RenderWorkerDevice
from generate.serializers import GeneratedImageSerializer
from model.models import TextToImageModel
from user.permissions import IsOwnerOrReadOnly

channel_layer = get_channel_layer()


class GeneratedImageViewSet(DynamicModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.

    Additionally, we will also provide an extra `generate` action in the future.
    """

    queryset = GeneratedImage.objects.all()
    serializer_class = GeneratedImageSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]

    # @action(detail=True, renderer_classes=[renderers.StaticHTMLRenderer])
    # def generate(self, request, *args, **kwargs):
    #    generated_image = self.get_object()
    #    return Response(generated_image.prompt)

    def perform_create(self, serializer: GeneratedImageSerializer) -> None:
        """Set the owner of the image to the current user."""
        serializer.save(owner=self.request.user)


def index(request: HttpRequest) -> HttpResponse:
    """
    The endpoint that returns the html UI used to generate images.

    Args:
        request: The request object.

    Returns:
        The rendered template.
    """
    models = TextToImageModel.objects.all().order_by("likes")

    context = {
        "use_localhost": "true" if settings.USE_LOCALHOST else "false",
        "models": models,
        "default_model": settings.DEFAULT_TEXT_TO_IMAGE_MODEL,
    }
    return render(request, "generate.html", context)


def queue_prompt(request: HttpRequest) -> JsonResponse:
    """Queue a prompt to be generated.

    Parses the request parameters and queues a celery task to generate an image.

    Args:
        request: The request object.

    Returns:
        A JSON response with an `error` key and a `task_id` key.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST request required", "task_id": None})
    from generate.tasks import generate_image as _generate_image

    guidance_scale = float(request.POST.get("guidance_scale", 7.5))
    inference_steps = int(request.POST.get("inference_steps", 50))
    height = int(request.POST.get("height", 512))
    width = int(request.POST.get("width", 512))
    seed = request.POST.get("seed", None)
    model: str = request.POST.get("model", settings.DEFAULT_TEXT_TO_IMAGE_MODEL)
    nsfw: bool = bool(len(request.POST.get("nsfw", "")))
    preview = bool(len(request.POST.get("use_preview", "")))
    prompt = request.POST["prompt"]
    negative_prompt = request.POST.get("negative_prompt", None)
    if seed:
        seed = int(seed)
    params = {
        "guidance_scale": guidance_scale,
        "num_inference_steps": inference_steps,
        "height": height,
        "width": width,
        "seed": seed,
        "nsfw": nsfw,
    }
    image = GeneratedImage(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        **params,
    )
    image.save()
    image_id = image.id
    models = model.split(";") if ";" in model else None
    result = _generate_image.apply_async(
        kwargs=dict(
            image_id=image_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_path_or_name=models or model,
            preview_image=preview,
            **params,
        ),
        countdown=2,
    )
    async_to_sync(channel_layer.group_send)(
        "generate",
        {"type": "event_message", "event": "image_queued", "message": str(image_id)},
    )
    return JsonResponse({"error": None, "image_id": image_id})


def csrf_form(request: HttpRequest) -> HttpResponse:
    """CSRF endpoint to get the middleware to set the token.

    Args:
        request: The request object.

    Returns:
        The rendered template.
    """
    return render(request, "csrf.html", {})


def images(
    request: HttpRequest, last: Optional[int] = None, page_size: int = 10
) -> JsonResponse:
    """Return generated images data as JSON.

    Args:
        request: The request object.
        last: The last image to return.
        page_size: The number of images to return.

    Returns:
        A JSON response with a list of generated image data in the `images` key,
        and a `total` key with the total number of images.
    """

    query = GeneratedImage.objects.all().order_by("-created_at")
    total = query.count()
    if last:
        generated_images = query[last * page_size : last * page_size + page_size]
    else:
        generated_images = query[:page_size]
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
                    "negative_prompt": image.negative_prompt,
                    "duration": f"{image.duration:.2f} seconds"
                    if image.duration
                    else None,
                    "seed": image.seed,
                    "guidance_scale": image.guidance_scale,
                    "num_inference_steps": image.num_inference_steps,
                    "height": image.height,
                    "width": image.width,
                    "model": image.model,
                    "error": image.error,
                }
                for image in generated_images
            ],
            "total": total,
        }
    )


def renderer_health(request: HttpRequest) -> JsonResponse:
    """Check health of all renderer devices.

    Args:
        request: The request object.

    Returns:
        A JSON response with a list of renderer device stats for the currently
          connected hosts by hostname.

        Each device reports the following stats:
        - id: The device ID.
        - name: The device name.
        - usage: The memory usage.
        - memory: The total memory.
        - device_allocated: The allocated memory.
        - device_cached: The cached memory.
    """
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


def renderer_status(request: HttpRequest) -> HttpResponse:
    """Render the renderer status page.

    Args:
        request: The request object.

    Returns:
        The rendered template.
        Displays a list of renderer devices by hostname.
    """
    return render(request, "status.html", {})
