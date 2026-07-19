#!/usr/bin/env python3
"""
Kaggle / local entry point — stage 1 (LoRA) and stage 2 (ControlNet) are independent.

Environment variables (optional):
  STAGE=lora|controlnet|all     # default: all
  MAX_TRAIN_SAMPLES=2000        # smoke test
  OUTPUT_DIR=/kaggle/working/outputs
  LORA_PATH=/kaggle/working/outputs/lora   # only used if present; stage 2 can omit
  CONTROLNET_MODEL_ID=lllyasviel/sd-controlnet-scribble
  DATASET_ID=...

Examples:
  # Train stage 2 first (no LoRA needed)
  STAGE=controlnet python train_all.py

  # Stage 1 only
  STAGE=lora python train_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/outputs"))
STAGE = os.getenv("STAGE", "all").lower()
MAX_SAMPLES = os.getenv("MAX_TRAIN_SAMPLES", "")
LORA_PATH = os.getenv("LORA_PATH", str(OUTPUT_DIR / "lora"))
DATASET_ID = os.getenv("DATASET_ID", "")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "")


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=TRAINING_DIR, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_args = ["--output_dir", str(OUTPUT_DIR)]
    if MAX_SAMPLES:
        base_args += ["--max_train_samples", MAX_SAMPLES]
    if DATASET_ID:
        base_args += ["--dataset_id", DATASET_ID]

    if STAGE in ("all", "lora"):
        run([sys.executable, "train_lora.py", *base_args])

    if STAGE in ("all", "controlnet"):
        cn_args = list(base_args)
        # LoRA is optional — only pass if the folder/file exists.
        lora = Path(LORA_PATH)
        if lora.exists():
            cn_args += ["--lora_path", str(lora)]
            print(f"[stage2] Using LoRA: {lora}", flush=True)
        else:
            print("[stage2] No LoRA found — fine-tuning ControlNet on base SD 1.5 UNet", flush=True)
        if CONTROLNET_MODEL_ID:
            cn_args += ["--controlnet_model_id", CONTROLNET_MODEL_ID]
        run([sys.executable, "train_controlnet.py", *cn_args])

    print("\nDone. Artifacts:")
    print(f"  LoRA:       {OUTPUT_DIR / 'lora'}")
    print(f"  ControlNet: {OUTPUT_DIR / 'controlnet'}")


if __name__ == "__main__":
    main()
