import os
import torch

# --- Base diffusion models ---
SD_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")

# Pretrained fallback: generic scribble ControlNet.
# After training, point to your Kaggle output, e.g.:
#   CONTROLNET_MODEL_ID=./models/controlnet
CONTROLNET_MODEL_ID = os.getenv(
    "CONTROLNET_MODEL_ID", "lllyasviel/sd-controlnet-scribble"
)

# Fashion LoRA from stage-1 training (optional but recommended).
#   FASHION_LORA_PATH=./models/lora
FASHION_LORA_PATH = os.getenv("FASHION_LORA_PATH", "")
FASHION_LORA_WEIGHT_NAME = os.getenv(
    "FASHION_LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors"
)

# LCM for fast 4-step inference (disable when using custom fashion weights).
USE_LCM = os.getenv("USE_LCM", "false").lower() in ("1", "true", "yes")
LCM_LORA_ID = os.getenv("LCM_LORA_ID", "latent-consistency/lcm-lora-sdv1-5")

# --- Device ---
DEVICE = os.getenv("DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Inference ---
RESOLUTION = int(os.getenv("RESOLUTION", "512" if DEVICE == "cuda" else "384"))

# Fashion model defaults (FashionSD-X paper); LCM uses fewer steps.
NUM_INFERENCE_STEPS = int(
    os.getenv("NUM_INFERENCE_STEPS", "4" if USE_LCM else "25")
)
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "1.0" if USE_LCM else "7.5"))
CONTROLNET_CONDITIONING_SCALE = float(
    os.getenv("CONTROLNET_CONDITIONING_SCALE", "0.6" if FASHION_LORA_PATH else "0.8")
)

# Convert freehand canvas to edge sketch before ControlNet (matches training data).
SKETCH_PREPROCESS = os.getenv("SKETCH_PREPROCESS", "adaptive")  # adaptive | canny | none

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

NEGATIVE_PROMPT = os.getenv(
    "NEGATIVE_PROMPT",
    (
        "person, human, man, woman, girl, boy, child, face, hands, arms, legs, body, "
        "model, posing, runway, street, room, furniture, mannequin head, "
        "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, text"
    ),
)
