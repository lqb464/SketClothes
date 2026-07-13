"""
Stage 2 — Train ControlNet (sketch + text → garment image).

Requires LoRA from stage 1 (recommended). ControlNet learns to follow sketch outlines
while the LoRA keeps outputs in the fashion/garment domain.

Output: outputs/controlnet/  (full diffusers ControlNet folder)

Usage:
  python train_controlnet.py --lora_path outputs/lora
  python train_controlnet.py --lora_path /kaggle/input/sketch2clothes-lora/lora
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from config import TrainConfig
from utils.dataset import DresscodeDataset, collate_controlnet, load_hf_dataset

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fashion ControlNet on sketch pairs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to stage-1 LoRA folder")
    parser.add_argument("--sd_model_id", type=str, default=None)
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--controlnet_train_steps", type=int, default=None)
    parser.add_argument("--controlnet_batch_size", type=int, default=None)
    parser.add_argument("--controlnet_gradient_accumulation", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    return parser.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    for key, value in vars(args).items():
        if value is None or key == "lora_path":
            continue
        if key == "output_dir":
            cfg.output_dir = Path(value)
        elif key == "learning_rate":
            cfg.controlnet_learning_rate = value
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(TrainConfig(), args)
    lora_path = Path(args.lora_path) if args.lora_path else None

    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.controlnet_gradient_accumulation,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(cfg.output_dir)),
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        cfg.controlnet_output_dir().mkdir(parents=True, exist_ok=True)
        logger.info("Initializing ControlNet from UNet: %s", cfg.sd_model_id)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.sd_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.sd_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd_model_id, subfolder="scheduler")

    controlnet = ControlNetModel.from_unet(unet)

    if lora_path and lora_path.exists():
        logger.info("Loading fashion LoRA into frozen UNet: %s", lora_path)
        unet.load_lora_weights(
            str(lora_path),
            weight_name="pytorch_lora_weights.safetensors",
        )
        unet.fuse_lora()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    hf_dataset = load_hf_dataset(
        cfg.dataset_id,
        cfg.dataset_split,
        cfg.max_train_samples,
    )
    train_dataset = DresscodeDataset(
        hf_dataset,
        tokenizer,
        resolution=cfg.resolution,
        center_crop=cfg.center_crop,
        random_flip=cfg.random_flip,
        with_control=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.controlnet_batch_size,
        shuffle=True,
        collate_fn=collate_controlnet,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=cfg.controlnet_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    num_update_steps = cfg.controlnet_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_update_steps,
    )

    controlnet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_loader, lr_scheduler
    )

    global_step = 0
    progress = tqdm(
        range(num_update_steps),
        disable=not accelerator.is_local_main_process,
        desc="ControlNet training",
    )

    while global_step < num_update_steps:
        for batch in train_loader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                control_values = batch["control_values"].to(dtype=weight_dtype)
                input_ids = batch["input_ids"]

                # Classifier-free guidance dropout on text + control (training trick).
                if cfg.controlnet_conditioning_dropout > 0:
                    drop_mask = (
                        torch.rand(pixel_values.shape[0], device=pixel_values.device)
                        < cfg.controlnet_conditioning_dropout
                    )
                    control_values = control_values.clone()
                    control_values[drop_mask] = 0.0

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_values,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = cfg.controlnet_output_dir() / f"checkpoint-{global_step}"
                    accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                    logger.info("Saved checkpoint: %s", save_path)

            if global_step >= num_update_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(cfg.controlnet_output_dir())
        logger.info("Saved ControlNet to %s", cfg.controlnet_output_dir())


if __name__ == "__main__":
    main()
