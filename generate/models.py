"""Models for the generate app."""
from django.db import models


# Create your models here.
class GeneratedImage(models.Model):
    """Model for generated images."""

    task_id = models.UUIDField()
    host = models.CharField(max_length=255, blank=True, null=True)
    duration = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    generated_at = models.DateTimeField(blank=True, null=True)
    prompt = models.TextField()
    seed = models.IntegerField(blank=True, null=True)
    guidance_scale = models.FloatField(blank=True, null=True)
    num_inference_steps = models.IntegerField(blank=True, null=True)
    height = models.IntegerField(blank=True, null=True)
    width = models.IntegerField(blank=True, null=True)

    def __str__(self):
        """Return the image path."""
        return f"{self.host}/media/{self.filename}"

    @property
    def filename(self):
        """Return the filename."""
        return f"{self.task_id}.png"
