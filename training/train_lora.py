"""
Stage 1 — LoRA fine-tune Stable Diffusion 1.5 on fashion garment images + captions.

Output: outputs/lora/pytorch_lora_weights.safetensors

Usage (local or Kaggle):
  python train_lora.py --output_dir /kaggle/working/outputs
  python train_lora.py --max_train_samples 2000 --lora_train_steps 1000  # smoke test
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
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from config import TrainConfig
from utils.dataset import DresscodeDataset, collate_lora, load_hf_dataset, split_train_val
from utils.validation import log_validation_lora

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fashion LoRA on Dress Code dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sd_model_id", type=str, default=None)
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_train_steps", type=int, default=None)
    parser.add_argument("--lora_batch_size", type=int, default=None)
    parser.add_argument("--lora_gradient_accumulation", type=int, default=None)
    parser.add_argument("--lora_text_dropout", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    return parser.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    for key, value in vars(args).items():
        if value is None:
            continue
        if key == "output_dir":
            cfg.output_dir = Path(value)
        elif key == "learning_rate":
            cfg.lora_learning_rate = value
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(TrainConfig(), args)

    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.lora_gradient_accumulation,
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(cfg.output_dir)),
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        cfg.lora_output_dir().mkdir(parents=True, exist_ok=True)
        logger.info("Loading base model: %s", cfg.sd_model_id)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.sd_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.sd_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd_model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Force float32 on CPU to prevent CPU mixed dtype crashes
    if accelerator.device.type == "cpu":
        weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    if weight_dtype == torch.float16:
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

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
        with_control=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.lora_batch_size,
        shuffle=True,
        collate_fn=collate_lora,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=cfg.lora_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    num_update_steps = cfg.lora_train_steps
    num_epochs = math.ceil(num_update_steps * cfg.lora_gradient_accumulation / len(train_loader))

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_update_steps,
    )

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    # Empty prompt embedding for CFG / text dropout (cached once).
    with torch.no_grad():
        empty_ids = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        empty_embed = text_encoder(empty_ids)[0]  # [1, seq, dim]

    if accelerator.is_main_process:
        logger.info(
            "LoRA text dropout=%.2f (generic/default captions always dropped)",
            cfg.lora_text_dropout,
        )

    global_step = 0
    progress = tqdm(
        range(num_update_steps),
        disable=not accelerator.is_local_main_process,
        desc="LoRA training",
    )

    unet.train()
    while global_step < num_update_steps:
        for batch in train_loader:
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
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

                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    # Generic captions → always unconditional; others → random dropout.
                    drop = batch["is_generic_caption"].to(device=latents.device)
                    if cfg.lora_text_dropout > 0:
                        drop = drop | (
                            torch.rand(bsz, device=latents.device) < cfg.lora_text_dropout
                        )
                    if drop.any():
                        uncond = empty_embed.to(dtype=encoder_hidden_states.dtype).expand(
                            bsz, -1, -1
                        )
                        encoder_hidden_states = torch.where(
                            drop.view(bsz, 1, 1),
                            uncond,
                            encoder_hidden_states,
                        )

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                    ).sample

                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % cfg.validation_steps == 0 and accelerator.is_main_process:
                    log_validation_lora(
                        accelerator.unwrap_model(unet),
                        vae,
                        text_encoder,
                        tokenizer,
                        noise_scheduler,
                        val_hf,
                        cfg.lora_output_dir(),
                        global_step,
                        accelerator.device,
                        weight_dtype,
                    )

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    _save_lora(unet, cfg.lora_output_dir() / f"checkpoint-{global_step}")

            if global_step >= num_update_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(unet, cfg.lora_output_dir())
        logger.info("Saved LoRA weights to %s", cfg.lora_output_dir())
        log_validation_lora(
            accelerator.unwrap_model(unet),
            vae,
            text_encoder,
            tokenizer,
            noise_scheduler,
            val_hf,
            cfg.lora_output_dir(),
            global_step,
            accelerator.device,
            weight_dtype,
        )


def _save_lora(unet, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet = unet.module if hasattr(unet, "module") else unet
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    from safetensors.torch import save_file

    save_file(lora_state_dict, output_dir / "pytorch_lora_weights.safetensors")


if __name__ == "__main__":
    main()
