"""
Generate text descriptions (captions) for garment photos using AI (BLIP or local VLM).
Used to prepare image-caption pairs for LoRA and ControlNet training when only photos are provided.

Usage:
  python training/utils/generate_captions.py
  python training/utils/generate_captions.py --batch_size 8 --save_txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI captions for garment photos.")
    parser.add_argument(
        "--photos_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "photos"),
        help="Path to photos directory."
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "captions.json"),
        help="Path to output JSON file storing all filename -> caption mappings."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace model ID for image captioning (default: Salesforce/blip-image-captioning-base)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generating captions."
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="If set, also saves individual .txt caption files inside captions folder."
    )
    parser.add_argument(
        "--txt_out_dir",
        type=str,
        default="",
        help="Directory to save individual .txt caption files (defaults to photos_dir.parent / 'captions', falling back to ./captions if read-only)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum length of generated caption tokens."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    photos_dir = Path(args.photos_dir)
    captions_file = Path(args.captions_file)

    if not photos_dir.exists():
        print(f"Error: Photos directory '{photos_dir}' does not exist.", flush=True)
        sys.exit(1)

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(photos_dir.glob(ext))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in {photos_dir}", flush=True)
        return

    print(f"[+] Found {len(image_paths)} images in {photos_dir}", flush=True)
    print(f"[+] Loading captioning model '{args.model_id}'...", flush=True)

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError:
        print("Error: transformers library not found or incomplete. Please run: pip install transformers", flush=True)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device.upper()}", flush=True)

    try:
        processor = BlipProcessor.from_pretrained(args.model_id)
        model = BlipForConditionalGeneration.from_pretrained(args.model_id).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{args.model_id}': {e}", flush=True)
        sys.exit(1)

    captions_dict = {}
    if captions_file.exists():
        try:
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            print(f"[+] Loaded {len(captions_dict)} existing captions from {captions_file}", flush=True)
        except Exception:
            captions_dict = {}

    if args.txt_out_dir:
        txt_out_dir = Path(args.txt_out_dir)
    else:
        # Fall back to local ./captions if photos_dir is in a read-only environment like Kaggle input
        parent_dir = photos_dir.parent
        if "kaggle/input" in str(parent_dir).lower() or "/input" in str(parent_dir).lower():
            txt_out_dir = Path("./captions")
            print(f"[!] Warning: photos_dir is in a read-only directory. Saving txt captions to local '{txt_out_dir}'", flush=True)
        else:
            txt_out_dir = parent_dir / "captions"

    if args.save_txt:
        txt_out_dir.mkdir(parents=True, exist_ok=True)

    # Try importing tqdm
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(image_paths), desc="Captioning")
    except ImportError:
        pbar = None

    batch_paths = []
    batch_images = []

    def process_batch(paths, images):
        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            captions = processor.batch_decode(out, skip_special_tokens=True)

            for path, cap in zip(paths, captions):
                clean_cap = cap.strip()
                # Store by relative filename (e.g., '000000.png') and by stem ('000000')
                captions_dict[path.name] = clean_cap
                captions_dict[path.stem] = clean_cap

                if args.save_txt:
                    txt_path = txt_out_dir / f"{path.stem}.txt"
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(clean_cap)
        except Exception as e:
            print(f"Error processing batch: {e}", flush=True)

    for idx, img_path in enumerate(image_paths):
        # Skip if already captioned
        if img_path.name in captions_dict and img_path.stem in captions_dict:
            if pbar:
                pbar.update(1)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            batch_paths.append(img_path)
            batch_images.append(img)
        except Exception as e:
            print(f"Error reading {img_path.name}: {e}", flush=True)
            if pbar:
                pbar.update(1)
            continue

        if len(batch_images) >= args.batch_size:
            process_batch(batch_paths, batch_images)
            if pbar:
                pbar.update(len(batch_images))
            elif (idx + 1) % 50 == 0:
                print(f"  Captioning progress: {idx + 1}/{len(image_paths)}...", flush=True)
            batch_paths = []
            batch_images = []

    if batch_images:
        process_batch(batch_paths, batch_images)
        if pbar:
            pbar.update(len(batch_images))

    if pbar:
        pbar.close()

    captions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Completed! Saved {len(captions_dict)} captions to '{captions_file}'", flush=True)
    if args.save_txt:
        print(f"[DONE] Also saved individual .txt files to '{txt_out_dir}'", flush=True)


if __name__ == "__main__":
    main()
