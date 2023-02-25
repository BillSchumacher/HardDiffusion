""" File and directory utilities. """
import glob
import os

import torch
from diffusers import DiffusionPipeline, __version__
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from huggingface_hub._snapshot_download import snapshot_download

from HardDiffusion.logs import logger

if SAFETENSORS_AVAILABLE := is_safetensors_available():
    logger.info("Safetensors is available.")
    import safetensors.torch


DEFAULT_NAMES = [
    WEIGHTS_NAME,
    SCHEDULER_CONFIG_NAME,
    CONFIG_NAME,
    ONNX_WEIGHTS_NAME,
    DiffusionPipeline.config_name,
]


def get_checkpoint_path(cached_path, attr):
    """Get the checkpoint path for the given attribute."""
    checkpoint_path = os.path.join(cached_path, attr)
    if os.path.exists(checkpoint_path):
        if files := [
            *glob.glob(os.path.join(checkpoint_path, "*.safetensors")),
            *glob.glob(os.path.join(checkpoint_path, "*.bin")),
        ]:
            return files[0]
    return None


def load_checkpoint(checkpoint_path):
    """Load the checkpoint from the given path."""
    return (
        safetensors.torch.load_file(checkpoint_path, device="cuda")
        if (SAFETENSORS_AVAILABLE and checkpoint_path.endswith(".safetensors"))
        else torch.load(checkpoint_path, map_location="cuda")
    )


def download_and_cache_models(
    pretrained_model_name_or_path_list,
    config_dicts,
    cache_dir,
    resume_download,
    proxies,
    local_files_only,
    revision,
):
    """Download and cache the models"""
    return [
        get_cached_folder(
            pretrained_model_name_or_path,
            cache_dir,
            config_dict,
            resume_download,
            proxies,
            local_files_only,
            revision,
        )
        for pretrained_model_name_or_path, config_dict in zip(
            pretrained_model_name_or_path_list, config_dicts
        )
    ]


def get_allowed_patterns(config_dict):
    """Get the allowed patterns for the given config dict."""
    folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
    allow_patterns = [os.path.join(k, "*") for k in folder_names]
    allow_patterns += DEFAULT_NAMES
    return allow_patterns


def get_cached_folder(
    pretrained_model_name_or_path,
    cache_dir,
    config_dict,
    resume_download,
    proxies,
    local_files_only,
    revision,
):
    """Get the cached folder for the given model."""
    requested_pipeline_class = config_dict.get("_class_name")
    user_agent = {
        "diffusers": __version__,
        "pipeline_class": requested_pipeline_class,
    }
    allow_patterns = get_allowed_patterns(config_dict)
    return (
        pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path)
        else snapshot_download(
            str(pretrained_model_name_or_path),
            cache_dir=cache_dir,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            revision=revision,
            allow_patterns=allow_patterns,
            user_agent=user_agent,
        )
    )
