import json
from datetime import datetime, timedelta

from django.shortcuts import redirect, render
from django.utils import timezone

from HardDiffusion.redthis import r
from model.models import (
    AudioClassificationModel,
    ConversationalModel,
    DepthEstimationModel,
    FeatureExtractionModel,
    ImageClassificationModel,
    ImageToImageModel,
    ImageToTextModel,
    ObjectDetectionModel,
    TextClassificationModel,
    TextGenerationModel,
    TextToImageModel,
    TextToSpeechModel,
    TokenClassificationModel,
    TranslationModel,
    VideoClassificationModel,
)


def add_model(request, model_class):
    model_id = request.GET.get("model_id")
    likes = request.GET.get("likes")
    if model_id:
        model = model_class(model_id=model_id, likes=int(likes))
        model.save()


def remove_model(request, model_class):
    model_id = request.GET.get("model_id")
    if model_id:
        model = model_class.objects.filter(model_id=model_id).first()
        if model:
            model.delete()


"""
pipeline_tag args.
 'Audio_to_Audio', 'AutomaticSpeechRecognition',
 'DocumentQuestionAnswering',  'Fill_Mask', 'GraphMachineLearning',
   'ImageSegmentation',  'QuestionAnswering',
     'ReinforcementLearning', 'Robotics', 'SentenceSimilarity', 'Summarization', 'TableQuestionAnswering',
       'TabularClassification', 'TabularRegression',
         'UnconditionalImageGeneration',
           'VisualQuestionAnswering', 'VoiceActivityDetection', 'Zero_ShotClassification', 'Zero_ShotImageClassification'

            'Translation', 'ObjectDetection', 
        'FeatureExtraction', 'Conversational', 'DepthEstimation', 'TokenClassification',
    'AudioClassification', 'ImageClassification', 'TextClassification',  'VideoClassification',
    'TextGeneration', 'Text_to_Image', 'Text_to_Speech',  'Image_to_Text', 'Image_to_Image',
"""


def add_huggingface_text_to_image_model(request):
    """Add text-to-image model."""
    add_model(request, TextToImageModel)
    return redirect("search_huggingface_text_to_image_models")


def remove_huggingface_text_to_image_model(request):
    """Remove text-to-image model."""
    remove_model(request, TextToImageModel)
    return redirect("search_huggingface_text_to_image_models")


def search_huggingface_text_to_image_models(request):
    return search_hugging_face_models(request, TextToImageModel, "text_to_image")


def add_huggingface_text_to_speech_model(request):
    """Add text-to-speech model."""
    add_model(request, TextToSpeechModel)
    return redirect("search_huggingface_text_to_speech_models")


def remove_huggingface_text_to_speech_model(request):
    """Remove text-to-speech model."""
    remove_model(request, TextToSpeechModel)
    return redirect("search_huggingface_text_to_speech_models")


def search_huggingface_text_to_speech_models(request):
    """Search text-to-speech models."""
    return search_hugging_face_models(request, TextToSpeechModel, "text_to_speech")


def add_huggingface_image_to_text_model(request):
    """Add image_to_text model."""
    add_model(request, ImageToTextModel)
    return redirect("search_huggingface_image_to_text_models")


def remove_huggingface_image_to_text_model(request):
    """Remove image_to_text model."""
    remove_model(request, ImageToTextModel)
    return redirect("search_huggingface_image_to_text_models")


def search_huggingface_image_to_text_models(request):
    """Search image_to_text models."""
    return search_hugging_face_models(request, ImageToTextModel, "image_to_text")


def add_huggingface_image_to_image_model(request):
    """Add image_to_image model."""
    add_model(request, ImageToImageModel)
    return redirect("search_huggingface_image_to_image_models")


def remove_huggingface_image_to_image_model(request):
    """Remove image_to_image model."""
    remove_model(request, ImageToImageModel)
    return redirect("search_huggingface_image_to_image_models")


def search_huggingface_image_to_image_models(request):
    """Search image_to_image models."""
    return search_hugging_face_models(request, ImageToImageModel, "image_to_image")


def add_huggingface_audio_classification_model(request):
    """Add audio_classification model."""
    add_model(request, AudioClassificationModel)
    return redirect("search_huggingface_audio_classification_models")


def remove_huggingface_audio_classification_model(request):
    """Remove audio_classification model."""
    remove_model(request, AudioClassificationModel)
    return redirect("search_huggingface_audio_classification_models")


def search_huggingface_audio_classification_models(request):
    """Search audio_classification models."""
    return search_hugging_face_models(
        request, AudioClassificationModel, "audio_classification"
    )


def add_huggingface_image_classification_model(request):
    """Add image_classification model."""
    add_model(request, ImageClassificationModel)
    return redirect("search_huggingface_image_classification_models")


def remove_huggingface_image_classification_model(request):
    """Remove image_classification model."""
    remove_model(request, ImageClassificationModel)
    return redirect("search_huggingface_image_classification_models")


def search_huggingface_image_classification_models(request):
    """Search image_classification models."""
    return search_hugging_face_models(
        request, ImageClassificationModel, "image_classification"
    )


def add_huggingface_text_classification_model(request):
    """Add text_classification model."""
    add_model(request, TextClassificationModel)
    return redirect("search_huggingface_text_classification_models")


def remove_huggingface_text_classification_model(request):
    """Remove text_classification model."""
    remove_model(request, TextClassificationModel)
    return redirect("search_huggingface_text_classification_models")


def search_huggingface_text_classification_models(request):
    """Search text_classification models."""
    return search_hugging_face_models(
        request, TextClassificationModel, "text_classification"
    )


def add_huggingface_token_classification_model(request):
    """Add token_classification model."""
    add_model(request, TokenClassificationModel)
    return redirect("search_huggingface_token_classification_models")


def remove_huggingface_token_classification_model(request):
    """Remove token_classification model."""
    remove_model(request, TokenClassificationModel)
    return redirect("search_huggingface_token_classification_models")


def search_huggingface_token_classification_models(request):
    """Search token_classification models."""
    return search_hugging_face_models(
        request, TokenClassificationModel, "token_classification"
    )


def add_huggingface_video_classification_model(request):
    """Add video_classification model."""
    add_model(request, VideoClassificationModel)
    return redirect("search_huggingface_video_classification_models")


def remove_huggingface_video_classification_model(request):
    """Remove video_classification model."""
    remove_model(request, VideoClassificationModel)
    return redirect("search_huggingface_video_classification_models")


def search_huggingface_video_classification_models(request):
    """Search video_classification models."""
    return search_hugging_face_models(
        request, VideoClassificationModel, "video_classification"
    )


def add_huggingface_text_generation_model(request):
    """Add text_generation model."""
    add_model(request, TextGenerationModel)
    return redirect("search_huggingface_text_generation_models")


def remove_huggingface_text_generation_model(request):
    """Remove text_generation model."""
    remove_model(request, TextGenerationModel)
    return redirect("search_huggingface_text_generation_models")


def search_huggingface_text_generation_models(request):
    """Search text_generation models."""
    return search_hugging_face_models(request, TextGenerationModel, "text_generation")


def add_huggingface_feature_extraction_model(request):
    """Add feature_extraction model."""
    add_model(request, FeatureExtractionModel)
    return redirect("search_huggingface_feature_extraction_models")


def remove_huggingface_feature_extraction_model(request):
    """Remove feature_extraction model."""
    remove_model(request, FeatureExtractionModel)
    return redirect("search_huggingface_feature_extraction_models")


def search_huggingface_feature_extraction_models(request):
    """Search feature_extraction models."""
    return search_hugging_face_models(
        request, FeatureExtractionModel, "feature_extraction"
    )


def add_huggingface_conversational_model(request):
    """Add conversational model."""
    add_model(request, ConversationalModel)
    return redirect("search_huggingface_conversational_models")


def remove_huggingface_conversational_model(request):
    """Remove conversational model."""
    remove_model(request, ConversationalModel)
    return redirect("search_huggingface_conversational_models")


def search_huggingface_conversational_models(request):
    """Search conversational models."""
    return search_hugging_face_models(request, ConversationalModel, "conversational")


def add_huggingface_depth_estimation_model(request):
    """Add depth_estimation model."""
    add_model(request, DepthEstimationModel)
    return redirect("search_huggingface_depth_estimation_models")


def remove_huggingface_depth_estimation_model(request):
    """Remove depth_estimation model."""
    remove_model(request, DepthEstimationModel)
    return redirect("search_huggingface_depth_estimation_models")


def search_huggingface_depth_estimation_models(request):
    return search_hugging_face_models(request, DepthEstimationModel, "depth_estimation")


def add_huggingface_translation_model(request):
    """Add translation model."""
    add_model(request, TranslationModel)
    return redirect("search_huggingface_translation_models")


def remove_huggingface_translation_model(request):
    """Remove translation model."""
    remove_model(request, TranslationModel)
    return redirect("search_huggingface_translation_models")


def search_huggingface_translation_models(request):
    """Search translation models."""
    return search_hugging_face_models(request, TranslationModel, "translation")


def add_huggingface_object_detection_model(request):
    """Add object_detection model."""
    add_model(request, ObjectDetectionModel)
    return redirect("search_huggingface_object_detection_models")


def remove_huggingface_object_detection_model(request):
    """Remove object_detection model."""
    remove_model(request, ObjectDetectionModel)
    return redirect("search_huggingface_object_detection_models")


def search_huggingface_object_detection_models(request):
    """Search object_detection models."""
    return search_hugging_face_models(request, ObjectDetectionModel, "object_detection")


def get_hugging_face_filter(model_type):
    from huggingface_hub import ModelFilter, ModelSearchArguments

    args = ModelSearchArguments()
    if model_type == "text_to_image":
        return ModelFilter(task=args.pipeline_tag.Text_to_Image)
    elif model_type == "text_to_speech":
        return ModelFilter(task=args.pipeline_tag.Text_to_Speech)
    elif model_type == "image_to_text":
        return ModelFilter(task=args.pipeline_tag.Image_to_Text)
    elif model_type == "image_to_image":
        return ModelFilter(task=args.pipeline_tag.Image_to_Image)
    elif model_type == "audio_classification":
        return ModelFilter(task=args.pipeline_tag.AudioClassification)
    elif model_type == "image_classification":
        return ModelFilter(task=args.pipeline_tag.ImageClassification)
    elif model_type == "text_classification":
        return ModelFilter(task=args.pipeline_tag.TextClassification)
    elif model_type == "token_classification":
        return ModelFilter(task=args.pipeline_tag.TokenClassification)
    elif model_type == "video_classification":
        return ModelFilter(task=args.pipeline_tag.VideoClassification)
    elif model_type == "text_generation":
        return ModelFilter(task=args.pipeline_tag.TextGeneration)
    elif model_type == "feature_extraction":
        return ModelFilter(task=args.pipeline_tag.FeatureExtraction)
    elif model_type == "conversational":
        return ModelFilter(task=args.pipeline_tag.Conversational)
    elif model_type == "depth_estimation":
        return ModelFilter(task=args.pipeline_tag.DepthEstimation)
    elif model_type == "translation":
        return ModelFilter(task=args.pipeline_tag.Translation)
    elif model_type == "object_detection":
        return ModelFilter(task=args.pipeline_tag.ObjectDetection)
    return None


def search_hugging_face_models(request, model_class, model_type):
    """Search models."""
    now = timezone.now()
    result = r.get("last_search")
    added_models = list(model_class.objects.values_list("model_id", flat=True))
    if not result:
        r.set("last_search", now.isoformat())
    else:
        last_search = datetime.fromisoformat(result.decode("utf-8"))
        if now - last_search < timedelta(days=1):
            if cached_models := r.get("models"):
                models = json.loads(cached_models.decode("utf-8"))
                for cached_model in models:
                    cached_model["added"] = cached_model["modelId"] in added_models
                return render(request, "search.html", {"models": models})
        else:
            r.set("last_search", now.isoformat())
    from huggingface_hub import HfApi

    api = HfApi()
    filter = get_hugging_face_filter(model_type)
    models = api.list_models(filter=filter)
    models = sorted(models, key=lambda x: x.likes)
    pipe = r.pipeline()
    pipe.set(
        f"{model_type}-models",
        json.dumps(
            [
                {
                    "modelId": model.modelId,
                    "likes": model.likes,
                    "lastModified": model.lastModified,
                    "tags": model.tags,
                    # 'downloads': model.downloads
                }
                for model in models
            ]
        ),
    )
    pipe.set("last_search", now.isoformat())
    for model in models:
        if model.modelId in added_models:
            setattr(model, "added", True)
        else:
            setattr(model, "added", False)
    pipe.execute()


"""
 ModelInfo: {
        modelId: raw-vitor/karoliel
        sha: 6f8ac2b15551561168a6c1555990695972e4a29f
        lastModified: 2023-02-15T19:22:22.000Z
        tags: ['diffusers', 'text-to-image', 'stable-diffusion', 'license:creativeml-openrail-m']
        pipeline_tag: text-to-image
        siblings: [RepoFile(rfilename='.gitattributes', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='README.md', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='feature_extractor/preprocessor_config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='karoliel.ckpt', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='model_index.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='safety_checker/config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='safety_checker/model.safetensors', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='safety_checker/pytorch_model.bin', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='scheduler/scheduler_config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='text_encoder/config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='text_encoder/pytorch_model.bin', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='tokenizer/merges.txt', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='tokenizer/special_tokens_map.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='tokenizer/tokenizer_config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='tokenizer/vocab.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='unet/config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='unet/diffusion_pytorch_model.bin', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='vae/config.json', size='None', blob_id='None', lfs='None'), RepoFile(rfilename='vae/diffusion_pytorch_model.bin', size='None', blob_id='None', lfs='None')]
        private: False
        author: raw-vitor
        config: None
        securityStatus: None
        _id: 63ed303ef3827af9bb4e6ed8
        id: raw-vitor/karoliel
        likes: 0
        library_name: diffusers
}
"""
