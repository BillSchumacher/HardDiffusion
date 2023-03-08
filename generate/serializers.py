""" Django Rest Framework Serializers for models in the generate app.
"""
from typing import Any, Dict

from dynamic_rest.serializers import DynamicModelSerializer
from rest_framework import serializers

from generate.models import GeneratedImage


class GeneratedImageSerializer(DynamicModelSerializer):
    """Serializer for GeneratedImage model."""

    filename = serializers.ReadOnlyField()

    class Meta:
        """Meta class for GeneratedImageSerializer."""

        model = GeneratedImage
        name = "generated_image"
        exclude = []

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
