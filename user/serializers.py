""" Django Rest Framework Serializers for models in the user app."""

from django.contrib.auth import get_user_model

from dynamic_rest.serializers import DynamicModelSerializer
from dynamic_rest.fields import DynamicRelationField


class UserSerializer(DynamicModelSerializer):
    """Serializer for User model."""
    generated_images = DynamicRelationField('GeneratedImageSerializer', many=True)

    class Meta:
        """Meta class for UserSerializer."""

        model = get_user_model()
        fields = ["id", "username", "generated_images"]
