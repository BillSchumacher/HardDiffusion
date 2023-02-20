from django.db import models


class ModelMixin(models.Model):
    """Model mixin for common fields."""

    model_id = models.CharField(max_length=255, blank=True, null=True)
    likes = models.IntegerField(blank=True, null=True)

    class Meta:
        abstract = True


class ConversationalModel(ModelMixin):
    pass


class DepthEstimationModel(ModelMixin):
    pass


class FeatureExtractionModel(ModelMixin):
    pass


class ObjectDetectionModel(ModelMixin):
    pass


class TextGenerationModel(ModelMixin):
    pass


class TranslationModel(ModelMixin):
    pass


class TextToImageModel(ModelMixin):
    pass


class TextToSpeechModel(ModelMixin):
    pass


class TextToTextModel(ModelMixin):
    pass


class ImageToImageModel(ModelMixin):
    pass


class ImageToTextModel(ModelMixin):
    pass


class AudioClassificationModel(ModelMixin):
    pass


class ImageClassificationModel(ModelMixin):
    pass


class TextClassificationModel(ModelMixin):
    pass


class TokenClassificationModel(ModelMixin):
    pass


class VideoClassificationModel(ModelMixin):
    pass
