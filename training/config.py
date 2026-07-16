"""Shared training configuration for Sketch2Clothes (FashionSD-X style)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # Base model — SD 1.5 fits Kaggle T4 (16GB) and matches FashionSD-X paper.
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"

    # HuggingFace dataset: image + text + sketch (Dress Code, ~48k rows).
    dataset_id: str = "Abhi5ingh/Dresscodepromptsketch"
    dataset_split: str = "train"

    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True

    # LoRA (stage 1) — text → garment image
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    )
    lora_learning_rate: float = 1e-4
    lora_train_steps: int = 10000
    lora_batch_size: int = 2
    lora_gradient_accumulation: int = 4

    # ControlNet (stage 2) — sketch + text → garment image
    # Fine-tune pretrained scribble ControlNet (NOT from_unet scratch).
    controlnet_model_id: str = "lllyasviel/sd-controlnet-scribble"
    controlnet_init_from_unet: bool = False  # True = train from scratch (legacy)
    controlnet_learning_rate: float = 1e-5
    controlnet_train_steps: int = 6000
    controlnet_batch_size: int = 1
    controlnet_gradient_accumulation: int = 8
    controlnet_conditioning_dropout: float = 0.05
    # Stage 1 LoRA is optional for stage 2 — omit --lora_path to freeze base UNet only.


    # Shared
    mixed_precision: str = "fp16"
    seed: int = 42
    checkpointing_steps: int = 500
    validation_steps: int = 500
    max_train_samples: int | None = None  # None = full dataset

    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def lora_output_dir(self) -> Path:
        return self.output_dir / "lora"

    def controlnet_output_dir(self) -> Path:
        return self.output_dir / "controlnet"

    def export_dir(self) -> Path:
        return self.output_dir / "export"
