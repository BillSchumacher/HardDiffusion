"""User models."""
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """User model."""

    huggingface_token = models.CharField(max_length=255, blank=True, null=True)
