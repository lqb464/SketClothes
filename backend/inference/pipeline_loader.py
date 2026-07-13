"""Build Stable Diffusion + ControlNet pipeline for inference."""

from __future__ import annotations

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

import config


def load_controlnet_pipeline(device: str, dtype: torch.dtype) -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_MODEL_ID,
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.SD_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )

    if config.FASHION_LORA_PATH:
        print(f"[INFO] Loading fashion LoRA: {config.FASHION_LORA_PATH}")
        pipe.load_lora_weights(
            config.FASHION_LORA_PATH,
            weight_name=config.FASHION_LORA_WEIGHT_NAME,
        )
        pipe.fuse_lora()

    if config.USE_LCM:
        print(f"[INFO] Loading LCM LoRA: {config.LCM_LORA_ID}")
        pipe.load_lora_weights(config.LCM_LORA_ID)
        pipe.fuse_lora()
        from diffusers import LCMScheduler

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)
