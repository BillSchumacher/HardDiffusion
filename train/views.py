from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.
@login_required
@user_passes_test(lambda u: u.is_superuser)
def index(request) -> HttpResponse:
    """Generate an image."""
    return render(request, "train.html")
