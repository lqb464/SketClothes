"""Dress Code sketch + caption dataset for LoRA and ControlNet training."""

from __future__ import annotations

import random
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_hf_dataset(
    dataset_id: str = "",
    split: str = "train",
    max_samples: int | None = None,
):
    import io
    import json
    from pathlib import Path
    import pandas as pd
    from PIL import Image
    from datasets import Dataset

    # Always check local data folder first
    data_dir = Path(__file__).resolve().parents[1] / "data"
    photos_dir = data_dir / "png" / "photos"
    sketches_dir = data_dir / "png" / "sketches"
    captions_file = data_dir / "png" / "captions.json"
    captions_dir = data_dir / "png" / "captions"

    # Support png, jpg, jpeg
    image_paths = []
    if photos_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            image_paths.extend(photos_dir.glob(ext))
        image_paths = sorted(list(set(image_paths)))

    # 1. Prioritize direct loading from local png/photos directory if available
    if image_paths:
        if max_samples is not None:
            image_paths = image_paths[:max_samples]

        captions_dict = {}
        if captions_file.exists():
            try:
                with open(captions_file, "r", encoding="utf-8") as f:
                    captions_dict = json.load(f)
            except Exception as e:
                print(f"Warning: could not read {captions_file}: {e}")

        default_prompt = "a high quality fashion garment photo, clean white background"

        def local_gen():
            for img_path in image_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Error reading image {img_path.name}: {e}")
                    continue

                # Locate corresponding sketch
                sketch_path = sketches_dir / f"{img_path.stem}.png"
                if not sketch_path.exists():
                    # Try exact filename matching
                    sketch_path = sketches_dir / img_path.name

                if sketch_path.exists():
                    try:
                        sketch = Image.open(sketch_path).convert("RGB")
                    except Exception as e:
                        print(f"Error reading sketch {sketch_path}: {e}")
                        sketch = Image.new("RGB", image.size, (255, 255, 255))
                else:
                    sketch = Image.new("RGB", image.size, (255, 255, 255))

                # Lookup caption (by exact filename, stem, or .txt file)
                text = captions_dict.get(img_path.name, captions_dict.get(img_path.stem, ""))
                if not text and captions_dir.exists():
                    txt_path = captions_dir / f"{img_path.stem}.txt"
                    if txt_path.exists():
                        try:
                            text = txt_path.read_text(encoding="utf-8").strip()
                        except Exception:
                            pass

                if not text:
                    text = default_prompt

                yield {
                    "image": image,
                    "text": text,
                    "sketch": sketch
                }

        return Dataset.from_generator(local_gen)

    # 2. Fallback to parquet loading if local png/photos folder is empty or non-existent
    parquet_dir = data_dir / "parquet" / "white-background"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Neither photos directory ({photos_dir}) nor Parquet directory ({parquet_dir}) found.")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    dfs = []
    for f in parquet_files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)

    if max_samples is not None:
        df = df.iloc[:min(max_samples, len(df))]

    def parquet_gen():
        for idx, row in df.iterrows():
            raw_img = row["image"]
            if isinstance(raw_img, dict):
                img_bytes = raw_img["bytes"]
            else:
                img_bytes = bytes(raw_img)

            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"Error reading image at index {idx}: {e}")
                continue

            sketch_path = sketches_dir / f"{idx:06d}.png"
            if sketch_path.exists():
                try:
                    sketch = Image.open(sketch_path).convert("RGB")
                except Exception as e:
                    print(f"Error reading sketch at {sketch_path}: {e}")
                    sketch = Image.new("RGB", image.size, (255, 255, 255))
            else:
                sketch = Image.new("RGB", image.size, (255, 255, 255))

            text = row.get("text", "")
            if not text:
                text = "a high quality fashion garment photo, clean white background"

            yield {
                "image": image,
                "text": text,
                "sketch": sketch
            }

    return Dataset.from_generator(parquet_gen)



class DresscodeDataset(Dataset):
    """Returns (pixel_values, input_ids) for LoRA or (+ control) for ControlNet."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        with_control: bool = False,
    ) -> None:
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.with_control = with_control

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.control_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize(self, caption: str) -> torch.Tensor:
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokens.input_ids.squeeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data[index]
        image: Image.Image = row["image"].convert("RGB")
        caption: str = row["text"]

        pixel_values = self.image_transforms(image)
        input_ids = self._tokenize(caption)

        sample = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "caption": caption,
        }

        if self.with_control:
            sketch: Image.Image = row["sketch"].convert("RGB")
            control_values = self.control_transforms(sketch)
            sample["control_values"] = control_values

        return sample


def collate_lora(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
    }


def collate_controlnet(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "control_values": torch.stack([item["control_values"] for item in batch]),
    }


def split_train_val(hf_dataset, val_ratio: float = 0.02, seed: int = 42):
    """Hold out a small validation split for checkpoint previews."""
    n = len(hf_dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return hf_dataset.select(train_indices), hf_dataset.select(val_indices)
