# coding=utf-8
# Copyright 2023 Bill Schumacher
#
# Some content in this file is adapted from the HuggingFace library.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib
import os
from packaging import version
from pathlib import Path
from typing import Optional, Union

import diffusers
import torch
import transformers
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BaseOutput,
    deprecate,
    get_class_from_dynamic_module,
    http_user_agent,
    is_accelerate_available,
    is_safetensors_available,
    is_torch_version,
    is_transformers_available,
)
from huggingface_hub import model_info, snapshot_download

from generate.pipeline_configuration import PipelineConfiguration
from HardDiffusion.logs import logger

if SAFETENSORS_AVAILABLE := is_safetensors_available():
    logger.info("Safetensors is available.")
    import safetensors.torch

if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel
else:
    raise ImportError("Transformers is not available. please install it.")

CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "diffusers.utils"
TRANSFORMERS_DUMMY_MODULES_FOLDER = "transformers.utils"

DEFAULT_NAMES = [
    WEIGHTS_NAME,
    SCHEDULER_CONFIG_NAME,
    CONFIG_NAME,
    ONNX_WEIGHTS_NAME,
    # DiffusionPipeline.config_name,
]


LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_pretrained", "from_pretrained"],
    },
}


ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def is_safetensors_compatible(info) -> bool:
    filenames = set(sibling.rfilename for sibling in info.siblings)
    pt_filenames = set(filename for filename in filenames if filename.endswith(".bin"))
    is_safetensors_compatible = any(file.endswith(".safetensors") for file in filenames)
    for pt_filename in pt_filenames:
        prefix, raw = os.path.split(pt_filename)
        if raw == "pytorch_model.bin":
            # transformers specific
            sf_filename = os.path.join(prefix, "model.safetensors")
        else:
            sf_filename = pt_filename[: -len(".bin")] + ".safetensors"
        if is_safetensors_compatible and sf_filename not in filenames:
            logger.warning(f"{sf_filename} not found")
            is_safetensors_compatible = False
    return is_safetensors_compatible


def from_pretrained(
    cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
):
    r"""
    Instantiate a PyTorch diffusion pipeline from pre-trained pipeline weights.

    The pipeline is set in evaluation mode by default using `model.eval()`
     (Dropout modules are deactivated).

    The warning *Weights from XXX not initialized from pretrained model* means
     that the weights of XXX do not come pretrained with the rest of the model.

    It is up to you to train those weights with a downstream fine-tuning task.

    The warning *Weights from XXX not used in YYY* means that the layer XXX is not used
     by YYY, therefore those weights are discarded.

    Parameters:
      pretrained_model_name_or_path (`str` or `os.PathLike`):
        Can be either:
          - A string, the *repo id* of a pretrained pipeline hosted inside
            a model repo on https://huggingface.co/ Valid repo ids have to be
            located under a user or organization name,
            like `CompVis/ldm-text2im-large-256`.
          - A path to a *directory* containing pipeline weights saved using
            [`~DiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.

      torch_dtype (`str` or `torch.dtype`, *optional*):
        Override the default `torch.dtype` and load the model under this dtype.
        If `"auto"` is passed the dtype will be automatically derived from the
         model's weights.

      custom_pipeline (`str`, *optional*):

        <Tip warning={true}>

            This is an experimental feature and is likely to change in the future.

        </Tip>

        Can be either:

            - A string, the *repo id* of a custom pipeline hosted inside a model repo on
                https://huggingface.co/.
              Valid repo ids have to be located under a user or organization name,
               like `hf-internal-testing/diffusers-dummy-pipeline`.

                <Tip>

                    It is required that the model repo has a file, called `pipeline.py`
                     that defines the custom pipeline.

                </Tip>

            - A string, the *file name* of a community pipeline hosted on GitHub under
                https://github.com/huggingface/diffusers/tree/main/examples/community.
              Valid file names have to match exactly the file name without `.py`
               located under the above link, *e.g.* `clip_guided_stable_diffusion`.

                <Tip>

                    Community pipelines are always loaded from the current `main`
                     branch of GitHub.

                </Tip>

            - A path to a *directory* containing a custom pipeline, e.g.,
               `./my_pipeline_directory/`.

                <Tip>

                    It is required that the directory has a file, called `pipeline.py`
                     that defines the custom pipeline.

                </Tip>

        For more information on how to load and create custom pipelines,
         please have a look at [Loading and Adding Custom Pipelines]
         (https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)

      torch_dtype (`str` or `torch.dtype`, *optional*):
      force_download (`bool`, *optional*, defaults to `False`):
        Whether or not to force the (re-)download of the model weights and
         configuration files, overriding the cached versions if they exist.
      resume_download (`bool`, *optional*, defaults to `False`):
        Whether or not to delete incompletely received files. Will attempt
         to resume the download if such a file exists.
      proxies (`Dict[str, str]`, *optional*):
        A dictionary of proxy servers to use by protocol or endpoint, e.g.,
         `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`.
         The proxies are used on each request.
      output_loading_info(`bool`, *optional*, defaults to `False`):
        Whether or not to also return a dictionary containing missing keys,
         unexpected keys and error messages.
      local_files_only(`bool`, *optional*, defaults to `False`):
        Whether or not to only look at local files
         (i.e., do not try to download the model).
      use_auth_token (`str` or *bool*, *optional*):
        The token to use as HTTP bearer authorization for remote files.
        If `True`, will use the token generated when running `huggingface-cli login`
         (stored in `~/.huggingface`).
      revision (`str`, *optional*, defaults to `"main"`):
        The specific model version to use. It can be a branch name, a tag name,
         or a commit id, since we use a git-based system for storing models and
         other artifacts on huggingface.co, so `revision` can be any identifier
         allowed by git.
      custom_revision (`str`, *optional*, defaults to `"main"` when loading from the
         Hub and to local version of `diffusers` when loading from GitHub):
        The specific model version to use. It can be a branch name, a tag name, or a
         commit id similar to `revision` when loading a custom pipeline from the Hub.
        It can be a diffusers version when loading a custom pipeline from GitHub.
      mirror (`str`, *optional*):
        Mirror source to accelerate downloads in China. If you are from China and have
         an accessibility problem, you can set this option to resolve it.
        Note that we do not guarantee the timeliness or safety.
        Please refer to the mirror site for more information.
        Specify the folder name here.
      device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
        A map that specifies where each submodule should go.
        It doesn't need to be refined to each parameter/buffer name, once a given
         module name is inside, every submodule of it will be sent to the same device.

        To have Accelerate compute the most optimized `device_map` automatically,
         set `device_map="auto"`.
        For more information about each option see [designing a device map]
        (https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
      low_cpu_mem_usage (`bool`, *optional*,
         defaults to `True` if torch version >= 1.9.0 else `False`):
        Speed up model loading by not initializing the weights and
         only loading the pre-trained weights. This also tries to not use more than 1x
         model size in CPU memory (including peak memory) while loading the model.
        This is only supported when torch version >= 1.9.0.
        If you are using an older version of torch, setting this argument to `True`
         will raise an error.
      return_cached_folder (`bool`, *optional*, defaults to `False`):
        If set to `True`, path to downloaded cached folder will be returned in addition
         to loaded pipeline.
      kwargs (remaining dictionary of keyword arguments, *optional*):
        Can be used to overwrite load - and saveable variables -
         *i.e.* the pipeline components - of the specific pipeline class.
        The overwritten components are then directly passed to the pipelines `__init__`
         method.
        See example below for more information.

    <Tip>

        It is required to be logged in (`huggingface-cli login`) when you want to use
         private or [gated models]
         (https://huggingface.co/docs/hub/models-gated#gated-models),
         *e.g.* `"runwayml/stable-diffusion-v1-5"`

    </Tip>

    <Tip>

    Activate the special ["offline-mode"]
     (https://huggingface.co/diffusers/installation.html#offline-mode) to use this
     method in a firewalled environment.

    </Tip>

    Examples:

    ```py
    >>> from diffusers import DiffusionPipeline

    >>> # Download pipeline from huggingface.co and cache.
    >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

    >>> # Download pipeline that requires an authorization token
    >>> # For more information on access tokens, please refer to this section
    >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
    >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    >>> # Use a different scheduler
    >>> from diffusers import LMSDiscreteScheduler

    >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    >>> pipeline.scheduler = scheduler
    ```
    """
    config = PipelineConfiguration(**kwargs)
    config.remove_config_keys(kwargs)

    # 1. Download the checkpoints and configs
    # use snapshot download here to get it working from from_pretrained
    if not os.path.isdir(pretrained_model_name_or_path):
        config_dict = cls.load_config(
            pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            resume_download=config.resume_download,
            force_download=config.force_download,
            proxies=config.proxies,
            local_files_only=config.local_files_only,
            use_auth_token=config.use_auth_token,
            revision=config.revision,
        )

        (
            allow_patterns,
            ignore_patterns,
            user_agent,
        ) = get_allow_and_ignore_patterns_with_user_agent(
            cls, pretrained_model_name_or_path, config_dict, config
        )
        # download all allow_patterns
        cached_folder = snapshot_download(
            pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            resume_download=config.resume_download,
            proxies=config.proxies,
            local_files_only=config.local_files_only,
            # use_auth_token=config.use_auth_token,
            revision=config.revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            user_agent=user_agent,
        )
    else:
        cached_folder = pretrained_model_name_or_path

    config_dict = cls.load_config(cached_folder)

    custom_pipeline = config.custom_pipeline
    # 2. Load the pipeline class, if using custom module then load it from the hub
    # if we load from explicit class, let's use it
    if custom_pipeline is not None:
        if custom_pipeline.endswith(".py"):
            path = Path(custom_pipeline)
            # decompose into folder & file
            file_name = path.name
            custom_pipeline = path.parent.absolute()
        else:
            file_name = CUSTOM_PIPELINE_FILE_NAME

        pipeline_class = get_class_from_dynamic_module(
            custom_pipeline,
            module_file=file_name,
            cache_dir=config.cache_dir,
            revision=config.custom_revision,
        )
    elif cls != DiffusionPipeline:
        pipeline_class = cls
    else:
        diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
        pipeline_class = getattr(diffusers_module, config_dict["_class_name"])

    # To be removed in 1.0.0
    if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
        version.parse(config_dict["_diffusers_version"]).base_version
    ) <= version.parse("0.5.1"):
        from diffusers import (
            StableDiffusionInpaintPipeline,
            StableDiffusionInpaintPipelineLegacy,
        )

        pipeline_class = StableDiffusionInpaintPipelineLegacy

        deprecation_message = (
            "You are using a legacy checkpoint for inpainting with Stable Diffusion,"
            f" therefore we are loading the {StableDiffusionInpaintPipelineLegacy}"
            f" class instead of {StableDiffusionInpaintPipeline}. For better inpainting"
            " results, we strongly suggest using Stable Diffusion's official inpainting"
            " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting"
            f" instead or adapting your checkpoint {pretrained_model_name_or_path} to"
            " the format of https://huggingface.co/runwayml/stable-diffusion-inpainting"
            ". Note that we do not actively maintain the"
            f" {StableDiffusionInpaintPipelineLegacy} class and will likely remove it"
            " in version 1.0.0."
        )
        deprecate(
            "StableDiffusionInpaintPipelineLegacy",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )

    # some modules can be passed directly to the init
    # in this case they are already instantiated in `kwargs`
    # extract them here
    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

    init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(
        config_dict, **kwargs
    )

    # define init kwargs
    init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
    init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

    # remove `null` components
    def load_module(name, value):
        if value[0] is None:
            return False
        if name in passed_class_obj and passed_class_obj[name] is None:
            return False
        return True

    init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

    if len(unused_kwargs) > 0:
        logger.warning(
            f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
        )

    if config.low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate`"
            " was not found in the environment. Defaulting to"
            " `low_cpu_mem_usage=False`. It is strongly recommended to install"
            " `accelerate` for faster and less memory-intense model loading."
            " You can do so with: \n```\npip install accelerate\n```\n."
        )

    if config.device_map is not None and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Loading and dispatching requires torch >= 1.9.0."
            " Please either update your PyTorch version or set"
            " `device_map=None`."
        )

    if config.low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0."
            " Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    if config.low_cpu_mem_usage is False and config.device_map is not None:
        raise ValueError(
            "You cannot set `low_cpu_mem_usage` to False while using"
            f" device_map={config.device_map} for loading and dispatching."
            " Please make sure to set `low_cpu_mem_usage=True`."
        )

    # import it here to avoid circular import
    from diffusers import pipelines

    # 3. Load each module in the pipeline
    for name, (library_name, class_name) in init_dict.items():
        # 3.1 - now that JAX/Flax is an official framework of the library,
        #  we might load from Flax names
        if class_name.startswith("Flax"):
            class_name = class_name[4:]

        is_pipeline_module = hasattr(pipelines, library_name)
        loaded_sub_model = None

        # if the model is in a pipeline module, then we load it from the pipeline
        if name in passed_class_obj:
            # 1. check that passed_class_obj has correct parent class
            if not is_pipeline_module:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {
                    c: getattr(library, c, None) for c in importable_classes.keys()
                }

                expected_class_obj = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(
                        class_obj, class_candidate
                    ):
                        expected_class_obj = class_candidate

                if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                    raise ValueError(
                        f"{passed_class_obj[name]} is of type:"
                        f" {type(passed_class_obj[name])}, but should be"
                        f" {expected_class_obj}"
                    )
            else:
                logger.warning(
                    f"You have passed a non-standard module {passed_class_obj[name]}."
                    " We cannot verify whether it has the correct type"
                )

            # set passed class object
            loaded_sub_model = passed_class_obj[name]
        elif is_pipeline_module:
            pipeline_module = getattr(pipelines, library_name)
            class_obj = getattr(pipeline_module, class_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            class_candidates = {c: class_obj for c in importable_classes.keys()}
        else:
            # else we just import it from the library.
            library = importlib.import_module(library_name)

            class_obj = getattr(library, class_name)
            importable_classes = LOADABLE_CLASSES[library_name]
            class_candidates = {
                c: getattr(library, c, None) for c in importable_classes.keys()
            }

        if loaded_sub_model is None:
            load_method_name = None
            for class_name, class_candidate in class_candidates.items():
                if class_candidate is not None and issubclass(
                    class_obj, class_candidate
                ):
                    load_method_name = importable_classes[class_name][1]

            if load_method_name is None:
                none_module = class_obj.__module__
                is_dummy_path = none_module.startswith(
                    DUMMY_MODULES_FOLDER
                ) or none_module.startswith(TRANSFORMERS_DUMMY_MODULES_FOLDER)
                if is_dummy_path and "dummy" in none_module:
                    # call class_obj for nice error message of missing requirements
                    class_obj()

                raise ValueError(
                    f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
                    f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
                )

            load_method = getattr(class_obj, load_method_name)
            loading_kwargs = {}

            if issubclass(class_obj, torch.nn.Module):
                loading_kwargs["torch_dtype"] = config.torch_dtype
            if issubclass(class_obj, diffusers.OnnxRuntimeModel):
                loading_kwargs["provider"] = config.provider
                loading_kwargs["sess_options"] = config.sess_options

            is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)
            is_transformers_model = (
                is_transformers_available()
                and issubclass(class_obj, PreTrainedModel)
                and version.parse(version.parse(transformers.__version__).base_version)
                >= version.parse("4.20.0")
            )

            # When loading a transformers model, if the device_map is None,
            #  the weights will be initialized as opposed to diffusers.
            # To make default loading faster we set the
            #  `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
            # This makes sure that the weights won't be initialized which significantly
            #  speeds up loading.
            if is_diffusers_model or is_transformers_model:
                loading_kwargs["device_map"] = config.device_map
                if config.from_flax:
                    loading_kwargs["from_flax"] = True

                # if `from_flax` and model is transformer model, can currently not load
                #  with `low_cpu_mem_usage`
                if not (config.from_flax and is_transformers_model):
                    loading_kwargs["low_cpu_mem_usage"] = config.low_cpu_mem_usage
                else:
                    loading_kwargs["low_cpu_mem_usage"] = False

            # check if the module is in a subdirectory
            if os.path.isdir(os.path.join(cached_folder, name)):
                loaded_sub_model = load_method(
                    os.path.join(cached_folder, name), **loading_kwargs
                )
            else:
                # else load from the root directory
                loaded_sub_model = load_method(cached_folder, **loading_kwargs)

        init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

    # 4. Potentially add passed objects if expected
    missing_modules = set(expected_modules) - set(init_kwargs.keys())
    passed_modules = list(passed_class_obj.keys())
    optional_modules = pipeline_class._optional_components
    if len(missing_modules) > 0 and missing_modules <= set(
        passed_modules + optional_modules
    ):
        for module in missing_modules:
            init_kwargs[module] = passed_class_obj.get(module, None)
    elif len(missing_modules) > 0:
        passed_modules = (
            set(list(init_kwargs.keys()) + list(passed_class_obj.keys()))
            - optional_kwargs
        )
        raise ValueError(
            f"Pipeline {pipeline_class} expected {expected_modules},"
            f" but only {passed_modules} were passed."
        )

    # 5. Instantiate the pipeline
    model = pipeline_class(**init_kwargs)

    if config.return_cached_folder:
        return model, cached_folder
    return model


def get_allow_and_ignore_patterns_with_user_agent(
    pipeline, pretrained_model_name_or_path, config_dict, pipeline_config
):
    # make sure we only download sub-folders and `diffusers` filenames
    folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
    allow_patterns = [os.path.join(k, "*") for k in folder_names]
    allow_patterns += DEFAULT_NAMES
    allow_patterns.append(pipeline.config_name)

    # make sure we don't download flax weights
    ignore_patterns = ["*.msgpack"]

    if pipeline_config.from_flax:
        ignore_patterns = ["*.bin", "*.safetensors"]
        allow_patterns += [
            SCHEDULER_CONFIG_NAME,
            CONFIG_NAME,
            FLAX_WEIGHTS_NAME,
            pipeline.config_name,
        ]

    custom_pipeline = pipeline_config.custom_pipeline
    if custom_pipeline is not None:
        allow_patterns += [CUSTOM_PIPELINE_FILE_NAME]
    clazz = pipeline.__class__

    if clazz != DiffusionPipeline:
        requested_pipeline_class = clazz.__name__
    else:
        requested_pipeline_class = config_dict.get("_class_name", clazz.__name__)
    user_agent = {"pipeline_class": requested_pipeline_class}
    if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
        user_agent["custom_pipeline"] = custom_pipeline

    user_agent = http_user_agent(user_agent)

    if is_safetensors_available() and not pipeline_config.local_files_only:
        info = model_info(
            pretrained_model_name_or_path,
            use_auth_token=pipeline_config.use_auth_token,
            revision=pipeline_config.revision,
        )
        if is_safetensors_compatible(info):
            ignore_patterns.append("*.bin")
        else:
            # as a safety mechanism we also don't download safetensors if
            # not all safetensors files are there
            ignore_patterns.append("*.safetensors")
    else:
        ignore_patterns.append("*.safetensors")
    return allow_patterns, ignore_patterns, user_agent
