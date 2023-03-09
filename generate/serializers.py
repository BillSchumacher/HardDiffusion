""" Django Rest Framework Serializers for models in the generate app.
"""
from typing import Any, Dict

from django.conf import settings

from dynamic_rest.serializers import DynamicModelSerializer
from rest_framework import serializers

from generate.models import GeneratedImage
from model.models import ImageToImageModel, TextToImageModel


class GeneratedImageSerializer(DynamicModelSerializer):
    """Serializer for GeneratedImage model."""

    filename = serializers.ReadOnlyField()
    owner = serializers.HiddenField(default=serializers.CurrentUserDefault())
    width = serializers.ChoiceField(
        choices=[(128, "128"), (256, "256"), (512, "512"), (768, "768"), (1024, "1024")]
    )
    height = serializers.ChoiceField(
        choices=[(128, "128"), (256, "256"), (512, "512"), (768, "768"), (1024, "1024")]
    )
    num_inference_steps = serializers.IntegerField(min_value=5, max_value=50)
    guidance_scale = serializers.FloatField(min_value=0.0, max_value=20.0)

    class Meta:
        """Meta class for GeneratedImageSerializer."""

        model = GeneratedImage
        name = "generated_image"
        exclude = []

    def validate_model(self, value: str) -> str:
        """Validate the model field.

        Args:
            value: (str) - the model repo id.

        Raises:
            ValidationError

        Returns:
            value: (str)
        """
        if not value:
            value = settings.DEFAULT_TEXT_TO_IMAGE_MODEL
        models = value.split(";") if ";" in value else []
        model_qs = TextToImageModel.objects
        if models:
            model_qs = model_qs.filter(model_id__in=models).all()
        else:
            model_qs = model_qs.filter(model_id=value).first()
        if not model_qs:
            raise serializers.ValidationError("Invalid model")
        return value

    def create(self, validated_data: Dict[str, Any]) -> GeneratedImage:
        """
        Create and return a new `GeneratedImages` instance, given the validated data.

        Args:
            validated_data (Dict[str, Any]): The validated data.

        Returns:
            GeneratedImage: The new GeneratedImage instance.
        """

        return GeneratedImage.objects.create(**validated_data)

    def update(
        self, instance: GeneratedImage, validated_data: Dict[str, Any]
    ) -> GeneratedImage:
        """
        Update and return an existing `GeneratedImage` instance, given the validated data.

        Args:
            instance (GeneratedImage): The existing GeneratedImage instance.
            validated_data (Dict[str, Any]): The validated data.

        Returns:
            GeneratedImage: The updated GeneratedImage instance.
        """
        instance.task_id = validated_data.get("task_id", instance.task_id)
        instance.batch_number = validated_data.get(
            "batch_number", instance.batch_number
        )
        instance.host = validated_data.get("host", instance.host)
        instance.duration = validated_data.get("duration", instance.duration)
        instance.created_at = validated_data.get("created_at", instance.created_at)
        instance.generated_at = validated_data.get(
            "generated_at", instance.generated_at
        )

        instance.prompt = validated_data.get("prompt", instance.prompt)
        instance.negative_prompt = validated_data.get(
            "negative_prompt", instance.negative_prompt
        )
        instance.seed = validated_data.get("seed", instance.seed)
        instance.guidance_scale = validated_data.get(
            "guidance_scale", instance.guidance_scale
        )
        instance.num_inference_steps = validated_data.get(
            "num_inference_steps", instance.num_inference_steps
        )
        instance.height = validated_data.get("height", instance.height)
        instance.width = validated_data.get("width", instance.width)
        instance.model = validated_data.get("model", instance.model)
        instance.error = validated_data.get("error", instance.error)
        instance.nsfw = validated_data.get("nsfw", instance.nsfw)

        instance.save()
        return instance
