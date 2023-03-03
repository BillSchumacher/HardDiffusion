"""User views."""
from django.contrib.auth import get_user_model

from rest_framework import viewsets

from user.serializers import UserSerializer


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    This viewset automatically provides `list` and `retrieve` actions.
    """

    queryset = get_user_model().objects.all()
    serializer_class = UserSerializer
