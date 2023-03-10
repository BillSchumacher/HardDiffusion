"""
ASGI config for HardDiffusion project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

import jwt
import rest_framework_simplejwt
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter


def encode(self, payload):
    """
    Returns an encoded token for the given payload dictionary.
    """
    jwt_payload = payload.copy()
    if self.audience is not None:
        jwt_payload["aud"] = self.audience
    if self.issuer is not None:
        jwt_payload["iss"] = self.issuer

    token = jwt.encode(jwt_payload, self.signing_key, algorithm=self.algorithm)
    if not isinstance(token, str):
        return token.decode("utf-8")
    return token


rest_framework_simplejwt.backends.TokenBackend.encode = encode
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
