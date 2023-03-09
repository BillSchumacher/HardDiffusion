"""Models for the generate app."""
from django.conf import settings
from django.db import models


# Create your models here.
class GeneratedImage(models.Model):
    """Model for generated images."""

    task_id = models.UUIDField(null=True)
    batch_number = models.IntegerField(blank=True, null=True)
    host = models.CharField(max_length=255, blank=True, null=True)
    duration = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    generated_at = models.DateTimeField(blank=True, null=True)
    prompt = models.TextField()
    negative_prompt: models.TextField = models.TextField(blank=True, null=True)
    seed = models.IntegerField(blank=True, null=True)
    guidance_scale = models.FloatField(blank=True, null=True)
    num_inference_steps = models.IntegerField(blank=True, null=True)
    height = models.IntegerField(blank=True, null=True)
    width = models.IntegerField(blank=True, null=True)
    model = models.CharField(max_length=255, blank=True, null=True)
    error = models.BooleanField(default=False)
    nsfw = models.BooleanField(default=False)
    private = models.BooleanField(default=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name="generated_images",
        on_delete=models.CASCADE,
        null=True,
    )

    def __str__(self):
        """Return the image path."""
        return f"{self.host}/media/{self.filename}"

    @property
    def filename(self):
        """Return the filename."""
        return f"{self.task_id}.png"


class RenderWorkerDevice(models.Model):
    """Model for render worker devices."""

    host = models.CharField(max_length=255, blank=True, null=True)
    device_id = models.IntegerField(blank=True, null=True)
    device_name = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_update_at = models.DateTimeField(auto_now=True)
    allocated_memory = models.FloatField(blank=True, null=True)
    total_memory = models.FloatField(blank=True, null=True)
    cached_memory = models.FloatField(blank=True, null=True)
