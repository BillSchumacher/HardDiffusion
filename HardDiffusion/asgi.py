"""
ASGI config for HardDiffusion project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HardDiffusion.settings")

django_asgi_app = get_asgi_application()

import generate.routing

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        # Requests from flutter have a parsed origin of None, so we can't
        # use the AllowedHostsOriginValidator here.
        "websocket": AuthMiddlewareStack(
            URLRouter(generate.routing.websocket_urlpatterns)
        ),
    }
)
