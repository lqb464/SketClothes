# Training pipeline — Sketch2Clothes

Train a **FashionSD-X style** model: SD 1.5 + fashion LoRA + sketch ControlNet.

Dataset: [`Abhi5ingh/Dresscodepromptsketch`](https://huggingface.co/datasets/Abhi5ingh/Dresscodepromptsketch) (~48k rows: `image`, `text`, `sketch`).

## Architecture

```
Stage 1 — train_lora.py
  Dress Code (image + caption) → LoRA on SD 1.5 UNet
  Output: outputs/lora/pytorch_lora_weights.safetensors

Stage 2 — train_controlnet.py
  Dress Code (sketch + caption → image) → ControlNet
  UNet frozen + fused with stage-1 LoRA
  Output: outputs/controlnet/

Inference (backend)
  sketch (canvas → edge) + text → SD 1.5 + LoRA + ControlNet → garment photo
```

## Local smoke test (CPU/GPU)

```bash
cd training
pip install -r requirements.txt

# Quick test (~2000 samples, fewer steps)
python train_lora.py --output_dir outputs --max_train_samples 500 --lora_train_steps 100
python train_controlnet.py --output_dir outputs --lora_path outputs/lora --max_train_samples 500 --controlnet_train_steps 100
```

## Kaggle (recommended)

### 1. Setup notebook

- **Accelerator:** GPU T4 x2 (or P100)
- **Internet:** ON (download SD 1.5 + dataset from HuggingFace)
- Upload this repo as Kaggle Dataset, or clone in notebook:

```python
!git clone https://github.com/YOUR_USER/Sketch2Clothes.git
%cd Sketch2Clothes/training
!pip install -q -r requirements.txt
```

### 2. Full training

```python
import os
os.environ["OUTPUT_DIR"] = "/kaggle/working/outputs"
# Optional smoke test first:
# os.environ["MAX_TRAIN_SAMPLES"] = "2000"

!python kaggle/train_all.py
```

Stages run sequentially (~4–8h on T4 for default steps).

### 3. Download artifacts

After run, download from **Output**:

| Path | Use |
|------|-----|
| `outputs/lora/` | Fashion LoRA weights |
| `outputs/controlnet/` | Sketch ControlNet |
| `outputs/export/sample.jpg` | Sanity-check image |

Upload as Kaggle Dataset for reuse, or copy to `backend/models/`.

### 4. Run only one stage

```python
os.environ["STAGE"] = "lora"        # or "controlnet"
os.environ["LORA_PATH"] = "/kaggle/input/your-lora-dataset/lora"
!python kaggle/train_all.py
```

## Default hyperparameters

| Param | LoRA | ControlNet |
|-------|------|------------|
| Steps | 8000 | 6000 |
| Batch | 2 | 1 |
| Grad accum | 4 | 8 |
| LR | 1e-4 | 1e-5 |
| Resolution | 512 | 512 |
| Rank | 64 | — |

Override via CLI, e.g. `--lora_train_steps 12000`.

## Connect trained model to backend

Copy weights to `backend/models/`:

```
backend/models/
├── lora/pytorch_lora_weights.safetensors
└── controlnet/   (config.json, diffusion_pytorch_model.safetensors, ...)
```

```powershell
cd backend
$env:CONTROLNET_MODEL_ID="./models/controlnet"
$env:FASHION_LORA_PATH="./models/lora"
$env:USE_LCM="false"
$env:SKETCH_PREPROCESS="adaptive"
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

## Train lại sau này

1. Thu sketch từ app → thêm vào dataset riêng
2. Fine-tune LoRA thêm vài epoch trên data mới
3. Fine-tune ControlNet trên sketch thật từ canvas (quan trọng nhất)
4. (Nâng cao) TexControl stage-2 texture refinement với ControlNet ip2p

## File map

```
training/
├── config.py              # Hyperparameters
├── train_lora.py          # Stage 1
├── train_controlnet.py    # Stage 2
├── train_all.py           # Kaggle orchestrator
└── utils/
    ├── dataset.py
    └── photos_to_sketches.py  # Convert photo dataset to sketches
```
