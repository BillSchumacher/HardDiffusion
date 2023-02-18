import json
from datetime import datetime, timedelta

from django.shortcuts import redirect, render
from django.utils import timezone

from HardDiffusion.redthis import r
from model.models import TextToImageModel

"""
pipeline_tag args.
'AudioClassification', 'Audio_to_Audio', 'AutomaticSpeechRecognition', 'Conversational', 'DepthEstimation',
 'DocumentQuestionAnswering', 'FeatureExtraction', 'Fill_Mask', 'GraphMachineLearning', 'ImageClassification',
   'ImageSegmentation', 'Image_to_Image', 'Image_to_Text', 'ObjectDetection', 'QuestionAnswering',
     'ReinforcementLearning', 'Robotics', 'SentenceSimilarity', 'Summarization', 'TableQuestionAnswering',
       'TabularClassification', 'TabularRegression', 'TextClassification', 'TextGeneration', 'Text_to_Image',
         'Text_to_Speech', 'TokenClassification', 'Translation', 'UnconditionalImageGeneration', 'VideoClassification',
           'VisualQuestionAnswering', 'VoiceActivityDetection', 'Zero_ShotClassification', 'Zero_ShotImageClassification'
"""


def add_huggingface_text_to_image_model(request):
    """Add text-to-image model."""
    model_id = request.GET.get("model_id")
    likes = request.GET.get("likes")
    if model_id:
        model = TextToImageModel(model_id=model_id, likes=int(likes))
        model.save()
    return redirect("search_huggingface_text_to_image_models")


def remove_huggingface_text_to_image_model(request):
    """Remove text-to-image model."""
    model_id = request.GET.get("model_id")
    if model_id:
        model = TextToImageModel.objects.filter(model_id=model_id).first()
        if model:
            model.delete()
    return redirect("search_huggingface_text_to_image_models")


def search_huggingface_text_to_image_models(request):
    """Search models."""
    now = timezone.now()
    result = r.get("last_search")
    added_models = list(TextToImageModel.objects.values_list("model_id", flat=True))
    if not result:
        r.set("last_search", now.isoformat())
    else:
        last_search = datetime.fromisoformat(result.decode("utf-8"))
        if now - last_search < timedelta(days=1):
            cached_models = r.get("models")
            if cached_models:
                models = json.loads(cached_models.decode("utf-8"))
                for cached_model in models:
                    if cached_model["modelId"] in added_models:
                        cached_model["added"] = True
                    else:
                        cached_model["added"] = False
                return render(request, "search.html", {"models": models})
        else:
            r.set("last_search", now.isoformat())

    from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

    api = HfApi()
    args = ModelSearchArguments()
    filter = ModelFilter(task=args.pipeline_tag.Text_to_Image)
    models = api.list_models(filter=filter)
    models = sorted(models, key=lambda x: x.likes)
    r.set(
        "models",
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
    r.set("last_search", now.isoformat())
    for model in models:
        if model.modelId in added_models:
            model["added"] = True
        else:
            model["added"] = False
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
