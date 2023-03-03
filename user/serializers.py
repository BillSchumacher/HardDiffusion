""" Django Rest Framework Serializers for models in the user app."""

from django.contrib.auth import get_user_model

from rest_framework import serializers

from generate.models import GeneratedImage


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model."""

    generated_images = serializers.PrimaryKeyRelatedField(
        many=True, queryset=GeneratedImage.objects.all()
    )

    class Meta:
        """Meta class for UserSerializer."""

        model = get_user_model()
        fields = ["id", "username", "generated_images"]
