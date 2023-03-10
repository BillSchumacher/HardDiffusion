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
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Q
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
from generate.tasks import generate_image as _generate_image
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
        model = serializer.validated_data.get("model")
        if not model:
            serializer.validated_data["model"] = settings.DEFAULT_TEXT_TO_IMAGE_MODEL
        obj = serializer.save(owner=self.request.user)
        model = obj.model
        models = model.split(";") if ";" in model else None
        image_id = obj.id
        preview = serializer.initial_data.get("preview")
        session_id = serializer.initial_data.get("session_id")
        params = {
            "guidance_scale": obj.guidance_scale,
            "num_inference_steps": obj.num_inference_steps,
            "height": obj.height,
            "width": obj.width,
            "seed": obj.seed,
            "nsfw": obj.nsfw,
        }
        result = _generate_image.apply_async(
            kwargs=dict(
                image_id=image_id,
                session_id=session_id,
                prompt=obj.prompt,
                negative_prompt=obj.negative_prompt,
                model_path_or_name=models or model,
                preview_image=preview,
                **params,
            ),
            countdown=2,
        )
        # TODO: Makes this happen at the same time as serializer save.
        obj.task_id = result.id
        obj.save()
        async_to_sync(channel_layer.group_send)(
            session_id,
            {
                "type": "event_message",
                "event": "image_queued",
                "message": str(result.id),
            },
        )

    def get_queryset(self, *args, **kwargs):
        if self.request.user.is_superuser:
            return GeneratedImage.objects.all()
        else:
            return GeneratedImage.objects.filter(
                Q(owner_id=self.request.user.id) | Q(private=False)
            )


@login_required
@user_passes_test(lambda u: u.is_superuser)
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
        "static_host": settings.STATIC_HOST,
    }
    return render(request, "generate.html", context)


def csrf_form(request: HttpRequest) -> HttpResponse:
    """CSRF endpoint to get the middleware to set the token.

    Args:
        request: The request object.

    Returns:
        The rendered template.
    """
    return render(request, "csrf.html", {})


@login_required
@user_passes_test(lambda u: u.is_superuser)
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


@login_required
@user_passes_test(lambda u: u.is_superuser)
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


@login_required
@user_passes_test(lambda u: u.is_superuser)
def renderer_status(request: HttpRequest) -> HttpResponse:
    """Render the renderer status page.

    Args:
        request: The request object.

    Returns:
        The rendered template.
        Displays a list of renderer devices by hostname.
    """
    return render(request, "status.html", {})
