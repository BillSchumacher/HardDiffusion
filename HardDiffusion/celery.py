""" celery app for HardDiffusion. """
import logging
import os
import sys

from celery import Celery, bootsteps

logger = logging.getLogger("HardDiffusion")


class WorkerHealthMonitor(bootsteps.StartStopStep):
    """
    This step registers a timer that runs a health check every 60 seconds.
    """

    requires = {"celery.worker.components:Timer", "celery.worker.components:Pool"}

    def __init__(self, _, **__):
        self.tref = None
        self.interval = 10

    def start(self, parent):
        logger.info(
            "Registering health monitor timer with %d seconds interval", self.interval
        )
        self.tref = parent.timer.call_repeatedly(
            self.interval,
            schedule_health_check,
            (parent,),
        )

    def stop(self, _):
        if self.tref:
            self.tref.cancel()
            self.tref = None


def schedule_health_check(worker):
    """Schedule a health check to be run in the render worker pool."""
    from generate.tasks import health_check

    worker.pool.apply_async(health_check)


# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "HardDiffusion.settings")
os.environ.setdefault("FORKED_BY_MULTIPROCESSING", "1")

app = Celery("HardDiffusion")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")
if "render_health" in sys.argv:
    app.steps["worker"].add(WorkerHealthMonitor)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()
