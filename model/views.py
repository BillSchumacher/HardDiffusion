import json
from datetime import datetime, timedelta
from typing import Union
from django.http import HttpResponsePermanentRedirect, HttpResponseRedirect

from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone

from huggingface_hub.hf_api import HfApi, ModelSearchArguments
from huggingface_hub.utils.endpoint_helpers import ModelFilter
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


HF_PIPELINE_TAG_TO_MODEL_CLASS = {
    "Audio_to_Audio": None,
    "AutomaticSpeechRecognition": None,
    "DocumentQuestionAnswering": None,
    "Fill_Mask": None,
    "GraphMachineLearning": None,
    "ImageSegmentation": None,
    "QuestionAnswering": None,
    "ReinforcementLearning": None,
    "Robotics": None,
    "SentenceSimilarity": None,
    "Summarization": None,
    "TableQuestionAnswering": None,
    "TabularClassification": None,
    "TabularRegression": None,
    "UnconditionalImageGeneration": None,
    "VisualQuestionAnswering": None,
    "VoiceActivityDetection": None,
    "Zero_ShotClassification": None,
    "Zero_ShotImageClassification": None,
    "Translation": TranslationModel,
    "ObjectDetection": ObjectDetectionModel,
    "FeatureExtraction": FeatureExtractionModel,
    "Conversational": ConversationalModel,
    "DepthEstimation": DepthEstimationModel,
    "TokenClassification": TokenClassificationModel,
    "AudioClassification": AudioClassificationModel,
    "ImageClassification": ImageClassificationModel,
    "TextClassification": TextClassificationModel,
    "VideoClassification": VideoClassificationModel,
    "TextGeneration": TextGenerationModel,
    "Text_to_Image": TextToImageModel,
    "Text_to_Speech": TextToSpeechModel,
    "Image_to_Text": ImageToTextModel,
    "Image_to_Image": ImageToImageModel,
}
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


def redirect_with_pipeline_tag(
    pipeline_tag: str
) -> Union[HttpResponseRedirect, HttpResponsePermanentRedirect]:
    """Redirect to search_hugging_face_models with pipeline_tag."""
    redirect_url = reverse('search_hugging_face_models')
    return redirect(f'{redirect_url}?pipeline_tag={pipeline_tag}')


def add_model(request, model_class):
    """Add model to database."""
    model_id = request.GET.get("model_id")
    likes = request.GET.get("likes")
    if model_id:
        model = model_class(model_id=model_id, likes=int(likes))
        model.save()


def remove_model(request, model_class):
    """Remove model from database."""
    if model_id := request.GET.get("model_id"):
        if model := model_class.objects.filter(model_id=model_id).first():
            model.delete()


def add_huggingface_model(
    request
) -> Union[HttpResponseRedirect, HttpResponsePermanentRedirect]:
    """Add object_detection model."""
    pipeline_tag = request.GET.get("pipeline_tag", "Text_to_Image")
    add_model(
        request,
        HF_PIPELINE_TAG_TO_MODEL_CLASS.get(pipeline_tag)
    )
    return redirect_with_pipeline_tag(pipeline_tag)


def remove_huggingface_model(
    request
) -> Union[HttpResponseRedirect, HttpResponsePermanentRedirect]:
    """Remove object_detection model."""
    pipeline_tag = request.GET.get("pipeline_tag", "Text_to_Image")
    remove_model(
        request,
        HF_PIPELINE_TAG_TO_MODEL_CLASS.get(pipeline_tag)
    )
    return redirect_with_pipeline_tag(pipeline_tag)


def search_huggingface_models(request):
    """Search object_detection models."""
    pipeline_tag = request.GET.get("pipeline_tag", "Text_to_Image")
    return search_hugging_face_models(
        request,
        HF_PIPELINE_TAG_TO_MODEL_CLASS.get(pipeline_tag),
        pipeline_tag
    )


def get_hf_filter(pipeline_tag) -> ModelFilter:
    """Get filter for pipeline_tag."""
    args = ModelSearchArguments()
    return ModelFilter(task=getattr(args.pipeline_tag, pipeline_tag))


def search_hugging_face_models(request, model_class, pipeline_tag):
    """Search models."""
    now = timezone.now()
    last_search_key = f"hf-{pipeline_tag}-last_search"
    models_key = f"hf-{pipeline_tag}-models"
    result = r.get(last_search_key)
    added_models = list(model_class.objects.values_list("model_id", flat=True))
    if not result:
        r.set(last_search_key, now.isoformat())
    else:
        last_search = datetime.fromisoformat(result.decode("utf-8"))
        if now - last_search < timedelta(days=1):
            if cached_models := r.get(models_key):
                models = json.loads(cached_models.decode("utf-8"))
                for cached_model in models:
                    cached_model["added"] = cached_model["modelId"] in added_models
                return render(request, "search.html", {"models": models})
        else:
            r.set(last_search_key, now.isoformat())

    api = HfApi()
    hf_filter = get_hf_filter(pipeline_tag)
    models = api.list_models(filter=hf_filter)
    models = sorted(models, key=lambda x: x.likes)
    pipe = r.pipeline()
    pipe.set(
        models_key,
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
    pipe.set(last_search_key, now.isoformat())
    for model in models:
        if model.modelId in added_models:
            setattr(model, "added", True)
        else:
            setattr(model, "added", False)
    pipe.execute()
    return render(request, "search.html", {"models": models})

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
