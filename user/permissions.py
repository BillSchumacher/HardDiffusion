"""Custom Django Rest Framework permissions for the user app."""
from typing import Any, Protocol

from django.db.models import Model
from django.http import HttpRequest

from rest_framework import permissions


class HasOwner(Protocol):
    """Protocol for objects with an owner attribute."""

    owner: Model


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object to edit it.
    """

    def has_object_permission(
        self, request: HttpRequest, _: Any, obj: HasOwner
    ) -> bool:
        """Check if the user is the owner of the object.

        Args:
            request: The request object.
            _: The view object.
            obj: The object to check.

        Returns:
            True if the user is the owner of the object, False otherwise.
        """
        # Read permissions are allowed to any request,
        # so we'll always allow GET, HEAD or OPTIONS requests.
        if request.method in permissions.SAFE_METHODS:
            return True

        # Write permissions are only allowed to the owner of the snippet.
        return obj.owner == request.user
