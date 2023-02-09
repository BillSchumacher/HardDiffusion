from django.urls import path
from generate import views

urlpatterns = [
    path('', views.index, name='generate_index'),
]
