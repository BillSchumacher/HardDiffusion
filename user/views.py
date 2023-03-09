"""User views."""
from django.contrib.auth import get_user_model

from dynamic_rest.viewsets import DynamicModelViewSet

from user.serializers import UserSerializer


class UserViewSet(DynamicModelViewSet):
    """
    This viewset automatically provides `list` and `retrieve` actions.
    """

    serializer_class = UserSerializer

    def get_queryset(self, *args, **kwargs):
        if self.request.user.is_superuser:
            return get_user_model().objects.all()
        else:
            return get_user_model().objects.filter(id=self.request.user.id)
