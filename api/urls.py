"""Django Rest Framework Router(s) for the API."""
from django.urls import include, path

from rest_framework.routers import DefaultRouter

from generate import views as generate_views
from user import views as user_views

# Create a router and register our viewsets with it.
router = DefaultRouter()

router.register(r"images", generate_views.GeneratedImageViewSet, basename="image")
router.register(r"users", user_views.UserViewSet, basename="user")

# The API URLs are now determined automatically by the router.
urlpatterns = [
    path("", include(router.urls)),
]