
"""Utilities for UNet models."""

from packaging import version

from diffusers.utils import deprecate

from generate.warnings import SAMPLE_SIZE_WARNING


def validate_unet_sample_size(unet, config, new_config):
    """Validate the UNet sample size."""
    is_unet_version_less_0_9_0 = hasattr(
        unet.config, "_diffusers_version"
    ) and version.parse(
        version.parse(config._diffusers_version).base_version
    ) < version.parse(
        "0.9.0.dev0"
    )
    is_unet_sample_size_less_64 = (
        hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
    )
    if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
        deprecate(
            "sample_size<64", "1.0.0", SAMPLE_SIZE_WARNING, standard_warn=False
        )
        new_config["sample_size"] = 64