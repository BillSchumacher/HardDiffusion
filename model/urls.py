"""model URL Configuration"""
from django.urls import path

from model import views

urlpatterns = [
    path(
        "",
        views.search_huggingface_text_to_image_models,
        name="search_huggingface_text_to_image_models",
    ),
    path(
        "add/text_to_image/huggingface",
        views.add_huggingface_text_to_image_model,
        name="add_huggingface_text_to_image_model",
    ),
    path(
        "remove/text_to_image/huggingface",
        views.remove_huggingface_text_to_image_model,
        name="remove_huggingface_text_to_image_model",
    ),
]
