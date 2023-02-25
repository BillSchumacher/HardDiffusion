"""Prompt utilities."""
from typing import List

import torch
from torch import FloatTensor

from HardDiffusion.logs import logger


def get_embed_from_prompt(tokenizer, text_encoder, prompt, device):
    """Get the embedding from the prompt."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because CLIP"
            " can only handle sequences up to %s tokens: %s",
            tokenizer.model_max_length,
            removed_text,
        )

    attention_mask = get_attention_mask(text_encoder, text_inputs, device)

    prompt_embeds = get_embeds_from_text_encoding(
        text_encoder, text_inputs, attention_mask, device
    )

    prompt_embeds = prompt_embeds[0]
    return prompt_embeds


def get_unconditional_embed(
    tokenizer, text_encoder, negative_prompt, prompt, prompt_embeds, batch_size, device
):
    """Get the embedding from the negative prompt."""
    uncond_tokens: List[str] = []
    if negative_prompt is None:
        uncond_tokens = [""] * batch_size
    elif type(prompt) is not type(negative_prompt):
        raise TypeError(
            f"`negative_prompt` should be the same type to `prompt`, "
            f"but got {type(negative_prompt)} != {type(prompt)}."
        )
    elif isinstance(negative_prompt, str):
        uncond_tokens = [negative_prompt]
    elif batch_size != len(negative_prompt):
        raise ValueError(
            f"`negative_prompt`: {negative_prompt} has batch size "
            f"{len(negative_prompt)}, but `prompt`: {prompt} has batch size "
            f"{batch_size}. Please make sure that passed `negative_prompt` matches"
            " the batch size of `prompt`."
        )
    else:
        uncond_tokens = negative_prompt

    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    attention_mask = get_attention_mask(text_encoder, uncond_input, device)

    negative_prompt_embeds = get_embeds_from_text_encoding(
        text_encoder, uncond_input, attention_mask, device
    )

    return negative_prompt_embeds[0]


def get_attention_mask(text_encoder, text_inputs, device):
    """Get the attention mask."""
    if (
        hasattr(text_encoder.config, "use_attention_mask")
        and text_encoder.config.use_attention_mask
    ):
        return text_inputs.attention_mask.to(device)
    return None


def get_embeds_from_text_encoding(text_encoder, text_inputs, attention_mask, device):
    """Get the embeddings from the text encoding."""
    return text_encoder(
        text_inputs.input_ids.to(device),
        attention_mask=attention_mask,
    )


def duplicate_embeddings(
    embeds, text_encoder, batch_size, num_images_per_prompt, device
) -> FloatTensor:
    """
    duplicate embeddings for each generation per prompt,
    using mps friendly method
    """
    seq_len = embeds.shape[1]
    embeds = embeds.to(dtype=text_encoder.dtype, device=device)
    embeds = embeds.repeat(1, num_images_per_prompt, 1)
    return embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
