"""Custom template filters for the HardDiffusion app."""
from django import template

register = template.Library()


def hf_model_name(value):
    """Get the model name from the HuggingFace model id."""
    return value.split("/")[1]


register.filter("hf_model_name", hf_model_name)
