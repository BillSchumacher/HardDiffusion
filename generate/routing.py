"""Websocket routing for the generate app."""
from django.urls import path

from generate import consumers


websocket_urlpatterns = [
    path("ws/generate/$", consumers.GenerateConsumer.as_asgi()),
]
