"""Django Rest Framework Router(s) for the API."""
from django.urls import include, path

from dynamic_rest.routers import DynamicRouter
from generate import views as generate_views
from user import views as user_views

# Create a router and register our viewsets with it.
router = DynamicRouter()

router.register(r"images", generate_views.GeneratedImageViewSet, base_name="image")
router.register(r"users", user_views.UserViewSet, base_name="user")

# The API URLs are now determined automatically by the router.
urlpatterns = [
    path("", include(router.urls)),
]
