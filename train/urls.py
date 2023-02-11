from django.urls import path

from train import views

urlpatterns = [
    path("", views.index, name="train_index"),
]
