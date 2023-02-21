"""model URL Configuration"""
from django.urls import path

from model import views

urlpatterns = [
    path(
        "",
        views.search_huggingface_models,
        name="search_huggingface_models",
    ),
    path(
        "add/model/huggingface",
        views.add_huggingface_model,
        name="add_huggingface_model",
    ),
    path(
        "remove/model/huggingface",
        views.remove_huggingface_model,
        name="remove_huggingface_model",
    ),
]
