#!/usr/bin/env python3
"""
Kaggle entry point — runs stage 1 (LoRA) then stage 2 (ControlNet).

In a Kaggle notebook, set GPU accelerator then run:
  !cd /kaggle/working/Sketch2Clothes/training && python kaggle/train_all.py

Environment variables (optional):
  STAGE=lora|controlnet|all
  MAX_TRAIN_SAMPLES=2000        # smoke test
  OUTPUT_DIR=/kaggle/working/outputs
  LORA_PATH=/kaggle/working/outputs/lora
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/outputs"))
STAGE = os.getenv("STAGE", "all").lower()
MAX_SAMPLES = os.getenv("MAX_TRAIN_SAMPLES", "")
LORA_PATH = os.getenv("LORA_PATH", str(OUTPUT_DIR / "lora"))


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=TRAINING_DIR, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_args = ["--output_dir", str(OUTPUT_DIR)]
    if MAX_SAMPLES:
        base_args += ["--max_train_samples", MAX_SAMPLES]

    if STAGE in ("all", "lora"):
        run([sys.executable, "train_lora.py", *base_args])

    if STAGE in ("all", "controlnet"):
        cn_args = base_args + ["--lora_path", LORA_PATH]
        run([sys.executable, "train_controlnet.py", *cn_args])

    print("\nDone. Download from Kaggle Output:")
    print(f"  LoRA:       {OUTPUT_DIR / 'lora'}")
    print(f"  ControlNet: {OUTPUT_DIR / 'controlnet'}")


if __name__ == "__main__":
    main()
