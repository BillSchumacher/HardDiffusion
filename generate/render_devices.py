""" Render devices. """
from django.conf import settings

import torch

from generate.models import RenderWorkerDevice


HOSTNAME = settings.HOSTNAME


def get_render_devices_by_id(device_count):
    """Get the existing render workers."""
    render_devices = RenderWorkerDevice.objects.filter(host=HOSTNAME).all()
    render_devices_by_id = {}
    new = False
    if not render_devices:
        new = True
        for device in range(device_count):
            device_name = torch.cuda.get_device_name(device)
            render_devices_by_id[device] = RenderWorkerDevice(
                device_id=device, host=HOSTNAME, device_name=device_name
            )
    else:
        for device in render_devices:
            render_devices_by_id[device.device_id] = device
    return render_devices, render_devices_by_id, new


def update_render_device_stats(device_id, render_device, now, new):
    """Update the render device stats."""
    device_free, device_memory = torch.cuda.mem_get_info(device_id)
    device_cached = torch.cuda.memory_cached(device_id)
    render_device.total_memory = device_memory
    render_device.allocated_memory = device_memory - device_free
    render_device.cached_memory = device_cached
    render_device.last_update_at = now
    if new:
        render_device.save()
