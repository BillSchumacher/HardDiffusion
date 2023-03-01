"""generate URL Configuration"""
from django.urls import path

from generate import views

urlpatterns = [
    path("", views.index, name="generate_index"),
    path("images/<int:last>/<int:page_size>", views.images, name="images"),
    path("images/<int:last>", views.images, name="images"),
    path("images", views.images, name="images"),
    path("queue_prompt", views.queue_prompt, name="queue_prompt"),
    path("renderer_health", views.renderer_health, name="renderer_health"),
    path("status", views.renderer_status, name="status"),
    path("csrf", views.csrf_form, name="csrf_form"),
]
