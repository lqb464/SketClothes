# Training pipeline — Sketch2Clothes

Train a **FashionSD-X style** model: SD 1.5 + fashion LoRA + sketch ControlNet.

Dataset: local `data/png/{photos,sketches}` or [`Abhi5ingh/Dresscodepromptsketch`](https://huggingface.co/datasets/Abhi5ingh/Dresscodepromptsketch).

## Two training tracks

| Track | Branch / path | What |
|-------|---------------|------|
| **Main** | `main` → `train_controlnet.py` | Fine-tune [`lllyasviel/sd-controlnet-scribble`](https://huggingface.co/lllyasviel/sd-controlnet-scribble) |
| **Side** | git branch `experiment/pix2pix` | Classic Pix2Pix (from [old project](https://github.com/lqb464/Sketch-to-Image-by-Pix2Pix)) |

Stage 1 (LoRA) and stage 2 (ControlNet) are **independent**. You can train stage 2 first without LoRA.

## Architecture (main track)

```
Stage 1 — train_lora.py   (optional, can run later)
  image + caption → LoRA on SD 1.5 UNet
  Output: outputs/lora/

Stage 2 — train_controlnet.py   (can run first)
  Fine-tune pretrained scribble ControlNet on sketch + caption → image
  UNet frozen (base SD, or + fused LoRA if --lora_path given)
  Output: outputs/controlnet/

Inference
  sketch + text → SD 1.5 (+ optional LoRA) + ControlNet → garment photo
```

## Train stage 2 first (recommended)

```bash
cd training
pip install -r requirements.txt

# Fine-tune scribble ControlNet — no LoRA required
python train_controlnet.py --output_dir outputs

# Smoke test
python train_controlnet.py --output_dir outputs --max_train_samples 500 --controlnet_train_steps 100
```

Optional later:

```bash
python train_lora.py --output_dir outputs
# Re-run stage 2 with LoRA fused into frozen UNet
python train_controlnet.py --output_dir outputs --lora_path outputs/lora
```

Legacy (train ControlNet from scratch — not recommended):

```bash
python train_controlnet.py --output_dir outputs --init_from_unet
```

## Kaggle

```python
import os
os.environ["OUTPUT_DIR"] = "/kaggle/working/outputs"
os.environ["STAGE"] = "controlnet"   # or "lora" / "all"
# os.environ["MAX_TRAIN_SAMPLES"] = "2000"
!python train_all.py
```

`train_all.py` only passes `--lora_path` if that folder exists.

## Default hyperparameters

| Param | LoRA | ControlNet (fine-tune) |
|-------|------|------------------------|
| Init | — | `lllyasviel/sd-controlnet-scribble` |
| Steps | 8000 | 6000 |
| Batch | 2 | 1 |
| Grad accum | 4 | 8 |
| LR | 1e-4 | 1e-5 |
| Resolution | 512 | 512 |

## Connect to backend

```
backend/models/
├── lora/pytorch_lora_weights.safetensors   # optional
└── controlnet/   (config.json, diffusion_pytorch_model.safetensors, ...)
```

```powershell
cd backend
$env:CONTROLNET_MODEL_ID="./models/controlnet"
$env:FASHION_LORA_PATH="./models/lora"
$env:USE_LCM="false"
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

## Pix2Pix side track

Branched after shared dataset prep (`c322383`, after `f3c34dd`):

```bash
git checkout experiment/pix2pix
cd training
python train_pix2pix.py --output_dir outputs/pix2pix
```

Files on that branch: `training/train_pix2pix.py`, `training/pix2pix_networks.py`.


## File map

```
training/
├── config.py
├── train_lora.py          # Stage 1 (optional)
├── train_controlnet.py    # Stage 2 main — fine-tune scribble
├── train_all.py
└── utils/
    ├── dataset.py
    └── photos_to_sketches.py
```
