"""Configuration utilities module for the pipeline."""
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE


class PipelineConfiguration:
    """Pipeline configuration class for the pipeline."""

    keys = [
        "cache_dir",
        "resume_download",
        "force_download",
        "proxies",
        "local_files_only",
        "from_flax",
        "custom_pipeline",
        "custom_revision",
        "provider",
        "sess_options",
        "low_cpu_mem_usage",
        "return_cached_folder",
        "use_auth_token",
        "revision",
        "torch_dtype",
        "device_map",
        "alpha",
        "interp",
    ]

    def __init__(
        self,
        cache_dir=DIFFUSERS_CACHE,
        resume_download: bool = False,
        force_download: bool = False,
        proxies=None,
        local_files_only: bool = HF_HUB_OFFLINE,
        from_flax: bool = False,
        custom_pipeline=None,
        custom_revision=None,
        provider=None,
        sess_options=None,
        low_cpu_mem_usage: bool = _LOW_CPU_MEM_USAGE_DEFAULT,
        return_cached_folder: bool = False,
        use_auth_token=None,
        revision=None,
        torch_dtype=None,
        device_map=None,
        alpha: float = 0.5,
        interp=None,
    ):
        self.cache_dir = cache_dir
        self.resume_download = resume_download
        self.force_download = force_download
        self.proxies = proxies
        self.local_files_only = local_files_only
        self.use_auth_token = use_auth_token
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.alpha = alpha
        self.interp = interp
        self.from_flax = from_flax
        self.custom_pipeline = custom_pipeline
        self.custom_revision = custom_revision
        self.provider = provider
        self.sess_options = sess_options
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.return_cached_folder = return_cached_folder

    def remove_config_keys(self, dict_):
        """
        Delete configuration keys from a dict.
        Optional modules are also passed in using kwargs.

        This function is used to cleanup the kwargs before passing them to the
        pipeline when loading a model.
        """
        for key in self.keys:
            if key in dict_:
                del dict_[key]
