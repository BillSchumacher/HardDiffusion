"""Scheduler utilities."""
from diffusers.utils import deprecate

from generate.warnings import (
    CLIP_SAMPLE_DEPRECATION_MESSAGE,
    STEPS_OFFSET_DEPRECATION_MESSAGE,
)


def validate_steps_offset(scheduler, config, new_config):
    """Validate the steps offset."""
    if hasattr(config, "steps_offset"):
        steps_offset = config.steps_offset
        if steps_offset == 1:
            return

        deprecate(
            "steps_offset!=1",
            "1.0.0",
            STEPS_OFFSET_DEPRECATION_MESSAGE.format(
                scheduler,
                steps_offset,
            ),
            standard_warn=False,
        )
        new_config["steps_offset"] = 1


def validate_clip_sample(scheduler, config, new_config):
    """Validate the clip sample."""
    if hasattr(config, "clip_sample"):
        if not config.clip_sample:
            return
        deprecate(
            "clip_sample not set",
            "1.0.0",
            CLIP_SAMPLE_DEPRECATION_MESSAGE.format(scheduler),
            standard_warn=False,
        )
        new_config["clip_sample"] = False
