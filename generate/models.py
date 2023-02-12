"""Models for the generate app."""
from django.db import models


# Create your models here.
class GeneratedImage(models.Model):
    """Model for generated images."""
    filename = models.CharField(max_length=255)
    host = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        """Return the image path."""
        return f"{self.host}/media/{self.filename}"
