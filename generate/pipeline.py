"""
The pipeline module.
"""

import glob
import inspect
import math
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline, __version__
from diffusers.configuration_utils import ConfigMixin, FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    load_pipeline_from_original_stable_diffusion_ckpt,
)
from diffusers.schedulers import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    DIFFUSERS_CACHE,
    ONNX_WEIGHTS_NAME,
    PIL_INTERPOLATION,
    WEIGHTS_NAME,
    deprecate,
    is_accelerate_available,
    is_safetensors_available,
    randn_tensor,
    replace_example_docstring,
)
from huggingface_hub._snapshot_download import snapshot_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from generate.input_validation import (
    validate_callback_steps,
    validate_generator_and_batch_size,
    validate_initial_image_latents,
    validate_negative_prompt_and_embeds,
    validate_prompt_and_embeds,
    validate_prompt_and_negative_embeds_shape,
    validate_prompt_type,
    validate_strength_range,
    validate_width_and_height,
)
from generate.noise import denoise
from generate.pipeline_doc_example import EXAMPLE_DOC_STRING
from generate.prompt import (
    duplicate_embeddings,
    get_embed_from_prompt,
    get_unconditional_embed,
)
from generate.schedulers import validate_clip_sample, validate_steps_offset
from generate.unet_utils import validate_unet_sample_size
from generate.warnings import SAFETY_CHECKER_WARNING
from HardDiffusion.logs import logger

if SAFETENSORS_AVAILABLE := is_safetensors_available():
    logger.info("Safetensors is available.")
    import safetensors.torch

if ACCELERATE_AVAILABLE := is_accelerate_available():
    logger.info("Accelerate is available.")
    from accelerate import cpu_offload
else:
    raise ImportError("Please install accelerate via `pip install accelerate`")


# Meta devices hold no data, this should be fine to keep in memory.
FAKE_DEVICE = torch.device("meta")


def validate_safety_checker(
    pipeline, safety_checker, feature_extractor, requires_safety_checker
):
    """Validate the safety checker."""
    clazz = pipeline.__class__
    if safety_checker is None and requires_safety_checker:
        logger.warning(SAFETY_CHECKER_WARNING, clazz)

    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            f"Make sure to define a feature extractor when loading {clazz}"
            " if you want to use the safety checker."
            " If you do not want to use the safety checker,"
            " you can pass `'safety_checker=None'` instead."
        )


def preprocess(image):
    """Preprocess the image for the model."""
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class HardDiffusionPipeline(DiffusionPipeline):
    """HardDiffusionPipeline is a pipeline that can be used to generate images
    from a given model or models.

    Currently a modified version of the custom_merged_pipeline is used.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[KarrasDiffusionSchedulers, SchedulerMixin, ConfigMixin],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        logger.info("HardDiffusionPipeline is being initialized.")
        super().__init__()

        # Validate the scheduler config
        scheduler_config = scheduler.config  # type: ignore
        new_scheduler_config = dict(scheduler_config)
        validate_steps_offset(scheduler, scheduler_config, new_scheduler_config)
        validate_clip_sample(scheduler, scheduler_config, new_scheduler_config)
        scheduler._internal_dict = FrozenDict(new_scheduler_config)  # type: ignore

        # Validate the safety checker
        validate_safety_checker(
            self, safety_checker, feature_extractor, requires_safety_checker
        )

        # Validate the unet config
        unet_config = unet.config
        new_unet_config = dict(unet_config)
        validate_unet_sample_size(unet, unet_config, new_unet_config)
        unet._internal_dict = FrozenDict(new_unet_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory
          usage. When called, unet, text_encoder, vae and safety checker have their
          state dicts saved to CPU and then are moved to a `torch.device('meta') and
          loaded to GPU only when their specific submodule has its `forward` method
          called.
        """

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker:
            cpu_offload(
                self.safety_checker, execution_device=device, offload_buffers=True
            )

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed.
          After calling `pipeline.enable_sequential_cpu_offload()` the execution
          device can only be inferred from Accelerate's module hooks.
        """
        if self.device != FAKE_DEVICE or not hasattr(self.unet, "_hf_hook"):
            return self.device
        unet_modules = self.unet.modules()
        return next(
            (
                torch.device(module._hf_hook.execution_device)
                for module in unet_modules
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                )
            ),
            self.device,
        )

    # Copied from diffusers StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
                Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, text embeddings will be generated from the `prompt` arg
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
                Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, negative_prompt_embeds will be generated from
                 `negative_prompt` input argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        tokenizer = self.tokenizer
        text_encoder = self.text_encoder

        if prompt_embeds is None:
            prompt_embeds = get_embed_from_prompt(
                tokenizer, text_encoder, prompt, device
            )

        prompt_embeds = duplicate_embeddings(
            prompt_embeds,
            text_encoder,
            prompt_embeds.shape[0],
            num_images_per_prompt,
            device,
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt_embeds = get_unconditional_embed(
                tokenizer,
                text_encoder,
                negative_prompt,
                prompt,
                prompt_embeds,
                batch_size,
                device,
            )

        if do_classifier_free_guidance:
            negative_prompt_embeds = duplicate_embeddings(
                negative_prompt_embeds,
                text_encoder,
                batch_size,
                num_images_per_prompt,
                device,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a
            # single batch to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        """run safety checker"""
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        """decode latents to image"""
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead
        #  and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        """
        prepare extra kwargs for the scheduler step, since not all schedulers have
        the same signature eta (η) is only used with the DDIMScheduler,
        it will be ignored for other schedulers.

        eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        and should be between [0, 1]
        """
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        strength,
        callback_steps,
        height=None,
        width=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """validate pipeline inputs"""
        validate_width_and_height(width, height)
        validate_prompt_type(prompt)
        validate_strength_range(strength)
        validate_callback_steps(callback_steps)
        validate_prompt_and_embeds(prompt, prompt_embeds)
        validate_negative_prompt_and_embeds(negative_prompt, negative_prompt_embeds)
        validate_prompt_and_negative_embeds_shape(prompt_embeds, negative_prompt_embeds)

    def get_timesteps(self, num_inference_steps, strength, device):
        """get the original timestep using init_timestep"""
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def _prepare_text_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """"prepare text-to-image latents"""
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        validate_generator_and_batch_size(generator, batch_size)

        latents = latents.to(device) if latents else randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )

        # scale the initial noise by the standard deviation required by the scheduler
        return latents * self.scheduler.init_noise_sigma

    def prepare_latents(
        self,
        image,
        height,
        width,
        num_channels_latents,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        """prepare latents"""
        batch_size = batch_size * num_images_per_prompt
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            return self._prepare_text_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generator,
                latents=latents,
            )
        image = image.to(device=device, dtype=dtype)

        validate_generator_and_batch_size(generator, batch_size)

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = 0.18215 * init_latents
        init_latents = validate_initial_image_latents(init_latents, batch_size)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        return self.scheduler.add_noise(init_latents, noise, timestep)


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, Image.Image] = None,
        strength: float = 0.8,
        width: Optional[int] = 512,
        height: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation.
                If not defined, one has to pass `prompt_embeds`. instead.

            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as
                the starting point for the process.

            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`.
                Must be between 0 and 1. `image` will be used as a starting point,
                 adding more noise to it the larger the `strength`.
                The number of denoising steps depends on the amount of noise initially
                 added.
                When `strength` is 1, added noise will be maximum and the denoising
                 process will run for the full number of iterations specified in
                 `num_inference_steps`.
                A value of 1, therefore, essentially ignores `image`.

            width (`int`, *optional*, defaults to 512): output image width

            height (`int`, *optional*, defaults to 512): output image height

            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the
                 expense of slower inference.
                This parameter will be modulated by `strength`.

            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance]
                (https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of
                 [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale > 1`.
                Higher guidance scale encourages to generate images that are closely
                 linked to the text `prompt`, usually at the expense of lower image
                 quality.

            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                If not defined, one has to pass `negative_prompt_embeds` instead.
                Ignored when not using guidance 
                (i.e., ignored if `guidance_scale` is less than `1`).

            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.

            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper:
                 https://arxiv.org/abs/2010.02502.
                Only applies to [`schedulers.DDIMScheduler`],
                 will be ignored for others.

            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)]
                 (https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.

            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
                Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, text embeddings will be generated from `prompt` input
                 argument.

            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
                Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, negative_prompt_embeds will be generated from
                 `negative_prompt` input argument.

            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                 [PIL](https://pillow.readthedocs.io/en/stable/):
                 `PIL.Image.Image` or `np.array`.

            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
                 instead of a plain tuple.

            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during
                 inference. The function will be called with the following arguments:
                 `callback(step: int, timestep: int, latents: torch.FloatTensor)`.

            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called.
                If not specified, the callback will be called at every step.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if
             `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated
             images, and the second element is a list of `bool`s denoting whether the
             corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        message = "Please use `image` instead of `init_image`."
        init_image = deprecate("init_image", "0.13.0", message, take_from=kwargs)
        image = init_image or image
        if image or init_image:
            width = None
            height = None
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of
        # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess image, if exists.
        if image:
            image = preprocess(image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if image:
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps, strength, device
            )
            num_channels_latents = 0
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        else:
            timesteps = self.scheduler.timesteps
            latent_timestep = None
            num_channels_latents = self.unet.in_channels

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            height,
            width,
            num_channels_latents,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = denoise(self.scheduler, self.unet,
                    latents, i, t, prompt_embeds, do_classifier_free_guidance,
                    guidance_scale, extra_step_kwargs, timesteps, num_warmup_steps,
                    progress_bar, callback, callback_steps
                )

        # 9. Post-processing
        image = self.decode_latents(latents)

        # 10. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return (
            StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )
            if return_dict
            else (image, has_nsfw_concept)
        )

    def _compare_model_configs(self, dict0, dict1):
        """Compares two model configs and returns True if they are the same."""
        if dict0 == dict1:
            return True
        config0, meta_keys0 = self._remove_meta_keys(dict0)
        config1, meta_keys1 = self._remove_meta_keys(dict1)
        if config0 == config1:
            print(f"Warning !: Mismatch in keys {meta_keys0} and {meta_keys1}.")
            return True
        return False

    def _remove_meta_keys(self, config_dict: Dict):
        """Remove the keys starting with '_' from the config dict"""
        meta_keys = []
        temp_dict = config_dict.copy()
        for key in config_dict.keys():
            if key.startswith("_"):
                temp_dict.pop(key)
                meta_keys.append(key)
        return (temp_dict, meta_keys)

    def validate_mergable_models(
        self,
        pretrained_model_name_or_path_list: List[Union[str, os.PathLike]],
        cache_dir,
        resume_download,
        force_download,
        proxies,
        local_files_only,
        use_auth_token,
        revision,
        force,
        **kwargs,
    ):
        """Validate that the checkpoints can be merged"""
        # Step 1: Load the model config and compare the checkpoints.
        #  We'll compare the model_index.json first while ignoring the keys
        #  starting with '_'
        config_dicts = [
            DiffusionPipeline.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
            for pretrained_model_name_or_path in pretrained_model_name_or_path_list
        ]

        comparison_result = True
        for idx in range(1, len(config_dicts)):
            comparison_result &= self._compare_model_configs(
                config_dicts[idx - 1], config_dicts[idx]
            )
            if not force and comparison_result is False:
                raise ValueError(
                    "Incompatible checkpoints."
                    " Please check model_index.json for the models."
                )
        print("Compatible model_index.json files found")
        return config_dicts

    def download_and_cache_models(
        self,
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

    @staticmethod
    def from_single_model(pretrained_model_name_or_path, **kwargs):
        """Create a single pipeline"""

        (
            cache_dir,
            resume_download,
            force_download,
            proxies,
            local_files_only,
            use_auth_token,
            revision,
            torch_dtype,
            device_map,
            _,
            _,
        ) = get_options_from_kwargs(kwargs)

        config_dict = DiffusionPipeline.load_config(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
        )
        cached_folder = get_cached_folder(
            pretrained_model_name_or_path,
            cache_dir,
            config_dict,
            resume_download,
            proxies,
            local_files_only,
            revision,
        )
        pipe = HardDiffusionPipeline.from_pretrained(
            cached_folder, torch_dtype=torch_dtype, device_map=device_map
        )
        pipe.to("cuda")
        return pipe

    def create_final_pipeline(
        self, cached_folders, torch_dtype, device_map, interp, alpha
    ):
        """Create the final pipeline"""
        # Step 3:-
        # Load the first checkpoint as a diffusion pipeline and modify its module
        #  state_dict in place
        final_pipe = HardDiffusionPipeline.from_pretrained(
            cached_folders[0], torch_dtype=torch_dtype, device_map=device_map
        )
        final_pipe.to("cuda")
        cached_folders_len = len(cached_folders)
        if cached_folders_len < 2:
            return final_pipe

        checkpoint_path_2 = None
        if len(cached_folders) > 2:
            checkpoint_path_2 = os.path.join(cached_folders[2])
            interp = "add_diff"

        if interp == "add_diff":
            theta_func = HardDiffusionPipeline.add_difference
        elif interp == "inv_sigmoid":
            theta_func = HardDiffusionPipeline.inv_sigmoid
        elif interp == "sigmoid":
            theta_func = HardDiffusionPipeline.sigmoid
        else:
            theta_func = HardDiffusionPipeline.weighted_sum

        # Find each module's state dict.
        for attr in final_pipe.config.keys():
            if attr.startswith("_"):
                continue
            checkpoint_path_1 = get_checkpoint_path(cached_folders[1], attr)
            if len(cached_folders) < 3:
                checkpoint_path_2 = None
            else:
                checkpoint_path_2 = get_checkpoint_path(cached_folders[2], attr)
            # For an attr if both checkpoint_path_1 and 2 are None, ignore.
            # If atleast one is present, deal with it according to interp method,
            # of course only if the state_dict keys match.
            if checkpoint_path_1 is None and checkpoint_path_2 is None:
                print(f"Skipping {attr}: not present in 2nd or 3rd model")
                continue
            try:
                module = getattr(final_pipe, attr)
                if isinstance(module, bool):  # ignore requires_safety_checker boolean
                    continue
                theta_0 = getattr(module, "state_dict")
                theta_0 = theta_0()

                update_theta_0 = getattr(module, "load_state_dict")
                theta_1 = load_checkpoint(checkpoint_path_1)
                theta_2 = None
                if checkpoint_path_2:
                    theta_2 = load_checkpoint(checkpoint_path_2)

                if theta_0.keys() != theta_1.keys():
                    print(f"Skipping {attr}: key mismatch")
                    continue
                if theta_2 and theta_1.keys() != theta_2.keys():
                    print(f"Skipping {attr}: key mismatch")
            except Exception as ex:
                print(f"Skipping {attr} do to an unexpected error: {str(ex)}")
                continue
            print(f"MERGING {attr}")

            for key in theta_0.keys():
                theta_0[key] = theta_func(
                    theta_0[key], theta_1[key], theta_2[key] if theta_2 else None, alpha
                )

            del theta_1
            del theta_2
            update_theta_0(theta_0)

            del theta_0
        return final_pipe

    @torch.no_grad()
    def merge(
        self,
        pretrained_model_name_or_path_list: List[Union[str, os.PathLike]],
        **kwargs,
    ):
        """
        Returns a new pipeline object of the class 'DiffusionPipeline' with the merged
        checkpoints(weights) of the models passed in the argument
        'pretrained_model_name_or_path_list' as a list.

        Parameters:
        -----------

         pretrained_model_name_or_path_list : A list of valid pretrained model names
         in the HuggingFace hub or paths to locally stored models in the
         HuggingFace format.

         **kwargs:
                Supports all the default DiffusionPipeline.get_config_dict kwargs viz..
                cache_dir, resume_download, force_download, proxies, local_files_only,
                  use_auth_token, revision, torch_dtype, device_map.
                alpha - The interpolation parameter. Ranges from 0 to 1.
                  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
                    would mean that the first model checkpoints would affect the final
                    result far less than an alpha of 0.2
                interp - The interpolation method to use for the merging.
                    Supports "sigmoid", "inv_sigmoid", "add_diff" and None.
                    Passing None uses the default interpolation which is weighted sum
                      interpolation.
                    For merging three checkpoints, only "add_diff" is supported.
                force - Whether to ignore mismatch in model_config.json
                    for the current models. Defaults to False.
        """
        # Default kwargs from DiffusionPipeline

        (
            cache_dir,
            resume_download,
            force_download,
            proxies,
            local_files_only,
            use_auth_token,
            revision,
            torch_dtype,
            device_map,
            alpha,
            interp,
        ) = get_options_from_kwargs(kwargs)

        print("Received list", pretrained_model_name_or_path_list)
        print(f"Combining with alpha={alpha}, interpolation mode={interp}")

        checkpoint_count = len(pretrained_model_name_or_path_list)
        # Ignore result from model_index_json comparision of the two checkpoints
        force = kwargs.pop("force", False)

        # If less than 2 checkpoints, nothing to merge.
        # If more than 3, not supported for now.
        if checkpoint_count > 3:
            raise ValueError(
                "Received incorrect number of checkpoints to merge."
                " Ensure that either 2 or 3 checkpoints are being passed."
            )

        print(f"Received the right number of checkpoints: {checkpoint_count}")
        # chkpt0, chkpt1 = pretrained_model_name_or_path_list[0:2]
        # chkpt2 = pretrained_model_name_or_path_list[2] \
        #  if checkpoint_count == 3 else None
        config_dicts = self.validate_mergable_models(
            pretrained_model_name_or_path_list,
            cache_dir,
            resume_download,
            force_download,
            proxies,
            local_files_only,
            use_auth_token,
            revision,
            force,
        )
        # Step 2: Basic Validation has succeeded.
        # Let's download the models and save them into our local files.
        cached_folders = self.download_and_cache_models(
            pretrained_model_name_or_path_list,
            config_dicts,
            cache_dir,
            resume_download,
            proxies,
            local_files_only,
            revision,
        )
        return self.create_final_pipeline(
            cached_folders, torch_dtype, device_map, interp, alpha
        )

    @staticmethod
    def weighted_sum(theta0, theta1, theta2, alpha):
        """Weighted sum interpolation."""
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def sigmoid(theta0, theta1, theta2, alpha):
        """Smoothstep interpolation."""
        alpha = alpha * alpha * (3 - (2 * alpha))
        return theta0 + ((theta1 - theta0) * alpha)

    # Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def inv_sigmoid(theta0, theta1, theta2, alpha):
        """Inverse smoothstep interpolation."""
        alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
        return theta0 + ((theta1 - theta0) * alpha)

    @staticmethod
    def add_difference(theta0, theta1, theta2, alpha):
        """Add the difference between theta1 and theta2 to theta0."""
        return theta0 + (theta1 - theta2) * (1.0 - alpha)


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


DEFAULT_NAMES = [
    WEIGHTS_NAME,
    SCHEDULER_CONFIG_NAME,
    CONFIG_NAME,
    ONNX_WEIGHTS_NAME,
    DiffusionPipeline.config_name,
]


def get_options_from_kwargs(kwargs):
    """Get the options from the kwargs."""
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    resume_download = kwargs.pop("resume_download", False)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    torch_dtype = kwargs.pop("torch_dtype", None)
    device_map = kwargs.pop("device_map", None)

    alpha = kwargs.pop("alpha", 0.5)
    interp = kwargs.pop("interp", None)

    return (
        cache_dir,
        resume_download,
        force_download,
        proxies,
        local_files_only,
        use_auth_token,
        revision,
        torch_dtype,
        device_map,
        alpha,
        interp,
    )


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


def get_pipeline(model_path_or_name, nsfw):
    """Get the pipeline for the given model path or name."""
    if isinstance(model_path_or_name, list):
        return HardDiffusionPipeline.from_pretrained(model_path_or_name[0])
    if model_path_or_name.startswith("./") and model_path_or_name.endswith(".ckpt"):
        return load_pipeline_from_original_stable_diffusion_ckpt(
            model_path_or_name,
            model_path_or_name.replace(".ckpt", ".yaml"),
        )
    # StableDiffusionPipeline.run_safety_checker = (
    #     run_safety_checker if nsfw else original_run_safety_checker
    # )
    # return StableDiffusionPipeline.from_pretrained(model_path_or_name)
    return HardDiffusionPipeline.from_single_model(model_path_or_name)
