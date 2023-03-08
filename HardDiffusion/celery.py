""" celery app for HardDiffusion. """
import logging
import os

from celery import Celery

logger = logging.getLogger("HardDiffusion")


# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HardDiffusion.settings")
os.environ.setdefault("FORKED_BY_MULTIPROCESSING", "1")

app = Celery("HardDiffusion")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")


# Load task modules from all registered Django apps.
app.autodiscover_tasks()
