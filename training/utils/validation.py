"""
Validation inference helpers for LoRA and ControlNet training.
Generates preview grids and side-by-side comparisons during training checkpoints.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def create_image_grid(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    """Create a grid of PIL images."""
    if not images:
        return Image.new("RGB", (512, 512), (255, 255, 255))
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        grid.paste(img, (idx % cols * w, idx // cols * h))
    return grid


def create_side_by_side_comparison(
    sketches: list[Image.Image],
    generated: list[Image.Image],
    ground_truths: list[Image.Image],
    resolution: int = 512,
) -> Image.Image:
    """Create a side-by-side comparison grid: [Sketch | Generated (ControlNet) | Ground Truth]."""
    num_samples = min(len(sketches), len(generated), len(ground_truths))
    if num_samples == 0:
        return Image.new("RGB", (resolution * 3, resolution), (255, 255, 255))

    header_height = 40
    canvas_w = resolution * 3
    canvas_h = header_height + num_samples * resolution

    composite = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 247))
    draw = ImageDraw.Draw(composite)

    # Draw header labels
    headers = ["Sketch Input", "Generated (ControlNet)", "Ground Truth Photo"]
    for col_idx, text in enumerate(headers):
        x = col_idx * resolution + 20
        draw.text((x, 12), text, fill=(30, 30, 30))

    for row_idx in range(num_samples):
        y = header_height + row_idx * resolution
        s_img = sketches[row_idx].resize((resolution, resolution), Image.Resampling.BILINEAR)
        g_img = generated[row_idx].resize((resolution, resolution), Image.Resampling.BILINEAR)
        t_img = ground_truths[row_idx].resize((resolution, resolution), Image.Resampling.BILINEAR)

        composite.paste(s_img, (0, y))
        composite.paste(g_img, (resolution, y))
        composite.paste(t_img, (resolution * 2, y))

    return composite


@torch.no_grad()
def log_validation_lora(
    unet: Any,
    vae: Any,
    text_encoder: Any,
    tokenizer: Any,
    scheduler: Any,
    val_dataset: Any,
    output_dir: Path,
    global_step: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    num_samples: int = 4,
) -> None:
    """Run validation inference for Stage 1 (LoRA) and save generated sample grid."""
    logger.info("Running validation inference for LoRA step %d...", global_step)
    was_training = unet.training
    unet.eval()

    try:
        # Take up to num_samples from validation set
        indices = list(range(min(num_samples, len(val_dataset))))
        if not indices:
            return

        prompts = [val_dataset[idx]["text"] for idx in indices]

        val_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=val_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)

        autocast_dtype = weight_dtype if weight_dtype in (torch.float16, torch.bfloat16) else torch.float32
        with torch.autocast(device.type if device.type != "mps" else "cpu", dtype=autocast_dtype):
            images = pipeline(
                prompt=prompts,
                num_inference_steps=20,
                generator=torch.Generator(device=device).manual_seed(42),
            ).images

        rows = 2 if len(images) > 2 else 1
        cols = (len(images) + rows - 1) // rows
        grid = create_image_grid(images, rows=rows, cols=cols)

        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        save_path = samples_dir / f"step-{global_step:06d}.png"
        grid.save(save_path)
        logger.info("Saved LoRA validation grid to %s", save_path)

    except Exception as e:
        logger.error("Error during LoRA validation inference: %s", e)
    finally:
        if was_training:
            unet.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def log_validation_controlnet(
    controlnet: Any,
    unet: Any,
    vae: Any,
    text_encoder: Any,
    tokenizer: Any,
    scheduler: Any,
    val_dataset: Any,
    output_dir: Path,
    global_step: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    num_samples: int = 4,
) -> None:
    """Run validation inference for Stage 2 (ControlNet) and save side-by-side comparison."""
    logger.info("Running validation inference for ControlNet step %d...", global_step)
    was_training = controlnet.training
    controlnet.eval()

    try:
        indices = list(range(min(num_samples, len(val_dataset))))
        if not indices:
            return

        sketches = []
        prompts = []
        ground_truths = []
        for idx in indices:
            row = val_dataset[idx]
            prompts.append(row["text"])
            sketches.append(row["sketch"].convert("RGB"))
            ground_truths.append(row["image"].convert("RGB"))

        val_scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=val_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)

        autocast_dtype = weight_dtype if weight_dtype in (torch.float16, torch.bfloat16) else torch.float32
        generated = []
        with torch.autocast(device.type if device.type != "mps" else "cpu", dtype=autocast_dtype):
            for prompt, sketch in zip(prompts, sketches):
                img = pipeline(
                    prompt=prompt,
                    image=sketch,
                    num_inference_steps=20,
                    generator=torch.Generator(device=device).manual_seed(42),
                ).images[0]
                generated.append(img)

        comparison = create_side_by_side_comparison(
            sketches=sketches,
            generated=generated,
            ground_truths=ground_truths,
        )

        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        save_path = samples_dir / f"step-{global_step:06d}.png"
        comparison.save(save_path)
        logger.info("Saved ControlNet validation comparison to %s", save_path)

    except Exception as e:
        logger.error("Error during ControlNet validation inference: %s", e)
    finally:
        if was_training:
            controlnet.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
