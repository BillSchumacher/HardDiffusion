"""Deprecation warnings for the generate module."""

CONFIG_FILE_DEPRECATION_MESSAGE = "The configuration file of this scheduler: %s "
DOWNLOADED_CHECKPOINT_MESSAGE = (
    "If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
    " nice if you could open a PR for the `scheduler/scheduler_config.json` file"
)

CLIP_SAMPLE_DEPRECATION_MESSAGE = (
    f"{CONFIG_FILE_DEPRECATION_MESSAGE}"
    "has not set the configuration `clip_sample`."
    " `clip_sample` should be set to False in the configuration file."
    " Please make sure to update the config accordingly as not setting `clip_sample`"
    " in the config might lead to incorrect results in future versions. "
    f"{DOWNLOADED_CHECKPOINT_MESSAGE}"
)

STEPS_OFFSET_DEPRECATION_MESSAGE = (
    f"{CONFIG_FILE_DEPRECATION_MESSAGE}"
    " is outdated."
    " `steps_offset` should be set to 1 instead of %s."
    " Please make sure to update the config accordingly as leaving `steps_offset` "
    "might led to incorrect results in future versions. "
    f"{DOWNLOADED_CHECKPOINT_MESSAGE}"
)

SAFETY_CHECKER_WARNING = (
    "You have disabled the safety checker for %s by passing `safety_checker=None`."
    " Ensure that you abide to the conditions of the Stable Diffusion license and"
    " do not expose unfiltered results in services or applications open to the public."
    " Both the diffusers team and Hugging Face strongly recommend to keep the safety"
    " filter enabled in all public facing circumstances, disabling it only for"
    " use-cases that involve analyzing network behavior or auditing its results."
    " For more information, please have a look at"
    " https://github.com/huggingface/diffusers/pull/254 ."
)

SAMPLE_SIZE_WARNING = (
    "The configuration file of the unet has set the default `sample_size` to smaller"
    " than 64 which seems highly unlikely."
    " If your checkpoint is a fine-tuned version of any of the following: \n"
    "- CompVis/stable-diffusion-v1-4 \n"
    "- CompVis/stable-diffusion-v1-3 \n"
    "- CompVis/stable-diffusion-v1-2 \n"
    "- CompVis/stable-diffusion-v1-1 \n"
    "- runwayml/stable-diffusion-v1-5\n"
    "- runwayml/stable-diffusion-inpainting \n\n"
    "You should change 'sample_size' to 64 in the configuration file."
    " Please make sure to update the config accordingly as leaving `sample_size=32`"
    " in the config might lead to incorrect results in future versions. "
    "If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
    " nice if you could open a Pull request for the `unet/config.json` file"
)
