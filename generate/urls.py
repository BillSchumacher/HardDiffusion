"""generate URL Configuration"""
from django.urls import path

from generate import views

urlpatterns = [
    path("", views.index, name="generate_index"),
    path("images", views.images, name="images"),
]
