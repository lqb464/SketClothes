"""
Stage 2 — Fine-tune ControlNet Scribble (sketch + text → garment image).

Independent of stage 1: LoRA is optional. Train stage 2 first with base SD 1.5,
then optionally fuse a fashion LoRA later for better garment style.

Default: fine-tune `lllyasviel/sd-controlnet-scribble` (has spatial prior).
Legacy: `--init_from_unet` trains ControlNet from scratch (slow, weak outline follow).

Output: outputs/controlnet/  (full diffusers ControlNet folder)

Usage:
  # Stage 2 only (recommended first)
  python train_controlnet.py --output_dir outputs

  # With stage-1 LoRA fused into frozen UNet
  python train_controlnet.py --output_dir outputs --lora_path outputs/lora
"""

from __future__ import annotations

import argparse
import logging
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
from utils.dataset import DresscodeDataset, collate_controlnet, load_hf_dataset, split_train_val
from utils.validation import log_validation_controlnet

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune fashion ControlNet on sketch pairs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional stage-1 LoRA folder (not required to train ControlNet)",
    )
    parser.add_argument(
        "--controlnet_model_id",
        type=str,
        default=None,
        help="Pretrained ControlNet to fine-tune (default: lllyasviel/sd-controlnet-scribble)",
    )
    parser.add_argument(
        "--init_from_unet",
        action="store_true",
        help="Train ControlNet from scratch via from_unet (legacy; ignore pretrained scribble)",
    )
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
    skip = {"lora_path", "init_from_unet"}
    for key, value in vars(args).items():
        if value is None or key in skip:
            continue
        if key == "output_dir":
            cfg.output_dir = Path(value)
        elif key == "learning_rate":
            cfg.controlnet_learning_rate = value
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    if args.init_from_unet:
        cfg.controlnet_init_from_unet = True
    return cfg


def load_optional_lora(unet: UNet2DConditionModel, lora_path: Path | None) -> None:
    if lora_path is None:
        logger.info("No --lora_path: training ControlNet against base SD UNet (stage 1 optional).")
        return
    if not lora_path.exists():
        logger.warning("LoRA path does not exist (%s) — continuing without LoRA.", lora_path)
        return

    logger.info("Loading fashion LoRA into frozen UNet: %s", lora_path)
    from safetensors.torch import load_file

    file_path = lora_path if lora_path.is_file() else lora_path / "pytorch_lora_weights.safetensors"
    state_dict = load_file(str(file_path))

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("base_model.model."):
            clean_state_dict[k[len("base_model.model.") :]] = v
        else:
            clean_state_dict[k] = v

    unet.load_attn_procs(clean_state_dict)
    unet.fuse_lora()


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

    tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.sd_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.sd_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd_model_id, subfolder="scheduler")

    if cfg.controlnet_init_from_unet:
        if accelerator.is_main_process:
            logger.warning(
                "Initializing ControlNet from UNet (scratch). Prefer fine-tuning %s",
                cfg.controlnet_model_id,
            )
        controlnet = ControlNetModel.from_unet(unet)
    else:
        if accelerator.is_main_process:
            logger.info("Fine-tuning pretrained ControlNet: %s", cfg.controlnet_model_id)
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model_id)

    load_optional_lora(unet, lora_path)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)
    controlnet.train()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.device.type == "cpu":
        weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=torch.float32)

    hf_dataset = load_hf_dataset(
        cfg.dataset_id,
        cfg.dataset_split,
        cfg.max_train_samples,
    )
    train_hf, val_hf = split_train_val(hf_dataset, val_ratio=0.01, seed=cfg.seed)
    logger.info("Split dataset: %d train samples, %d validation samples", len(train_hf), len(val_hf))
    train_dataset = DresscodeDataset(
        train_hf,
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
    # Shorter warmup when fine-tuning a pretrained ControlNet.
    warmup_steps = 100 if not cfg.controlnet_init_from_unet else 500
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    controlnet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_loader, lr_scheduler
    )

    global_step = 0
    progress = tqdm(
        range(num_update_steps),
        disable=not accelerator.is_local_main_process,
        desc="ControlNet fine-tune",
    )

    while global_step < num_update_steps:
        for batch in train_loader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                control_values = batch["control_values"].to(dtype=weight_dtype)
                input_ids = batch["input_ids"]

                if cfg.controlnet_conditioning_dropout > 0:
                    drop_mask = (
                        torch.rand(pixel_values.shape[0], device=pixel_values.device)
                        < cfg.controlnet_conditioning_dropout
                    )
                    control_values = control_values.clone()
                    control_values[drop_mask] = 0.0

                with accelerator.autocast():
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

                if global_step % cfg.validation_steps == 0 and accelerator.is_main_process:
                    log_validation_controlnet(
                        accelerator.unwrap_model(controlnet),
                        unet,
                        vae,
                        text_encoder,
                        tokenizer,
                        noise_scheduler,
                        val_hf,
                        cfg.controlnet_output_dir(),
                        global_step,
                        accelerator.device,
                        weight_dtype,
                    )

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
        log_validation_controlnet(
            accelerator.unwrap_model(controlnet),
            unet,
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            val_hf,
            cfg.controlnet_output_dir(),
            global_step,
            accelerator.device,
            weight_dtype,
        )


if __name__ == "__main__":
    main()
