"""Input validation for the pipeline."""
from typing import Optional, Union

import torch
from diffusers.utils import deprecate


def validate_width_and_height(width: Optional[int], height: Optional[int]):
    """Validate that width and height are divisible by 8."""
    if height and width and (height % 8 != 0 or width % 8 != 0):
        raise ValueError(
            "`height` and `width` have to be divisible by 8 but are: "
            f"{height} and {width}."
        )


def validate_prompt_type(prompt: Union[str, list]):
    """Validate that the prompt type is valid."""
    if not isinstance(prompt, str) and not isinstance(prompt, list):
        raise ValueError(
            f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
        )


def validate_strength_range(strength: Union[float, int]):
    """Validate that the strength is in the range [0.0, 1.0]."""
    if strength < 0 or strength > 1:
        raise ValueError(
            f"The value of strength should in [0.0, 1.0] but is {strength}"
        )


def validate_callback_steps(callback_steps):
    """Validate that the callback steps are a positive integer."""
    if (
        callback_steps is None
        or not isinstance(callback_steps, int)
        or callback_steps <= 0
    ):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is "
            f"{callback_steps} of type {type(callback_steps)}."
        )


def validate_prompt_and_embeds(prompt, prompt_embeds):
    """Validate that the prompt and prompt_embeds are valid."""
    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and "
            f"`prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. "
            "Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (
        not isinstance(prompt, str) and not isinstance(prompt, list)
    ):
        raise ValueError(
            f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
        )


def validate_negative_prompt_and_embeds(negative_prompt, negative_prompt_embeds):
    """Validate that the negative prompt and negative prompt_embeds are valid."""
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and"
            f" `negative_prompt_embeds`: {negative_prompt_embeds}. "
            "Please make sure to only forward one of the two."
        )


def validate_prompt_and_negative_embeds_shape(prompt_embeds, negative_prompt_embeds):
    """Validate that the prompt and negative prompt embeds have the same shape."""
    if (
        prompt_embeds is not None
        and negative_prompt_embeds is not None
        and prompt_embeds.shape != negative_prompt_embeds.shape
    ):
        raise ValueError(
            "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when"
            " passed directly, but got: "
            f"`prompt_embeds` {prompt_embeds.shape} != "
            f"`negative_prompt_embeds` {negative_prompt_embeds.shape}."
        )


def validate_generator_and_batch_size(generator, batch_size):
    """
    Validate that the generator is a list and has the same length as the batch size.
    """
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)},"
            f" but requested an effective batch size of {batch_size}."
            " Make sure the batch size matches the length of the generators."
        )


def validate_initial_image_latents(init_latents, batch_size):
    """Validate that the initial image latents and batch size are valid."""
    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`),"
            f" but only {init_latents.shape[0]} initial images (`image`). "
            "Initial images are now duplicated to match the number of text prompts."
            " Note that this behavior is deprecated and will be removed in a"
            " version 1.0.0. Please make sure to update your script to pass "
            "as many initial images as text prompts to suppress this warning."
        )
        deprecate(
            "len(prompt) != len(image)",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0]:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]}"
            f" to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)
    return init_latents
