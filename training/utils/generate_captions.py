"""
Generate text descriptions (captions) for garment photos.

Default model: microsoft/Florence-2-large (DETAILED_CAPTION) — much more stable
than BLIP-base on fashion product shots (avoids repetition loops).

Usage:
  python training/utils/generate_captions.py --save_txt --force
  python training/utils/generate_captions.py --model_id Salesforce/blip-image-captioning-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from caption_utils import is_degenerate_caption, normalize_product_caption

DEFAULT_MODEL = "microsoft/Florence-2-large"
FLORENCE_TASK = "<DETAILED_CAPTION>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI captions for garment photos.")
    parser.add_argument(
        "--photos_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "photos"),
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "captions.json"),
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HF model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (Florence-2: keep 1–2 on T4).",
    )
    parser.add_argument("--save_txt", action="store_true")
    parser.add_argument("--txt_out_dir", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recaption all images (also overwrites degenerate existing captions).",
    )
    parser.add_argument(
        "--fix_bad_only",
        action="store_true",
        help="Only recaption missing or degenerate captions; keep good ones.",
    )
    return parser.parse_args()


def _is_florence(model_id: str) -> bool:
    return "florence" in model_id.lower()


def _ensure_flash_attn_optional() -> None:
    """Florence-2 remote code may hard-require flash_attn; stub it so eager/sdpa works on Kaggle."""
    try:
        import flash_attn  # noqa: F401

        return
    except Exception:
        pass

    import types

    stub = types.ModuleType("flash_attn")
    stub.__version__ = "2.5.0"
    sys.modules["flash_attn"] = stub
    for name in (
        "flash_attn.flash_attn_interface",
        "flash_attn.bert_padding",
        "flash_attn.layers",
        "flash_attn.layers.rotary",
    ):
        mod = types.ModuleType(name)
        sys.modules[name] = mod

    # Satisfy transformers availability checks without a real CUDA wheel.
    try:
        import transformers.utils.import_utils as import_utils

        import_utils._flash_attn_2_available = True  # type: ignore[attr-defined]
        import_utils.is_flash_attn_2_available = lambda: True  # type: ignore[assignment]
        import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True  # type: ignore[assignment]
    except Exception:
        pass


def _patch_florence_config_compat() -> None:
    """Newer transformers removed auto attrs like forced_bos_token_id; Florence remote code still reads them."""
    try:
        from transformers.configuration_utils import PretrainedConfig

        _orig_getattr = PretrainedConfig.__getattribute__

        def _safe_getattr(self, key):  # type: ignore[no-untyped-def]
            try:
                return _orig_getattr(self, key)
            except AttributeError:
                if key in {
                    "forced_bos_token_id",
                    "forced_eos_token_id",
                    "force_bos_token_to_be_generated",
                }:
                    return None
                raise

        PretrainedConfig.__getattribute__ = _safe_getattr  # type: ignore[method-assign]
    except Exception:
        pass


def _load_florence(model_id: str, device: str):
    _ensure_flash_attn_optional()
    _patch_florence_config_compat()
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # Prefer eager/sdpa — flash_attn wheels rarely build on Kaggle.
    model = None
    last_err: Exception | None = None
    for attn_impl in ("eager", "sdpa", None):
        try:
            kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"Failed to load Florence-2: {last_err}")
    model = model.to(device)
    model.eval()
    return processor, model, "florence"


def _load_blip(model_id: str, device: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, "blip"


def _caption_florence(processor, model, images, device: str, max_new_tokens: int) -> list[str]:
    dtype = next(model.parameters()).dtype
    outs: list[str] = []
    for image in images:
        inputs = processor(text=FLORENCE_TASK, images=image, return_tensors="pt")
        inputs = {
            k: (v.to(device, dtype=dtype) if torch.is_floating_point(v) else v.to(device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        raw = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            raw,
            task=FLORENCE_TASK,
            image_size=(image.width, image.height),
        )
        text = parsed.get(FLORENCE_TASK, "")
        if isinstance(text, dict):
            text = text.get("caption") or str(text)
        outs.append(str(text).strip())
    return outs


def _caption_blip(processor, model, images, device: str, max_new_tokens: int) -> list[str]:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
            repetition_penalty=1.2,
        )
    return [c.strip() for c in processor.batch_decode(out, skip_special_tokens=True)]


def main() -> None:
    args = parse_args()
    photos_dir = Path(args.photos_dir)
    captions_file = Path(args.captions_file)

    if not photos_dir.exists():
        print(f"Error: Photos directory '{photos_dir}' does not exist.", flush=True)
        sys.exit(1)

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths: list[Path] = []
    for ext in image_extensions:
        image_paths.extend(photos_dir.glob(ext))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in {photos_dir}", flush=True)
        return

    print(f"[+] Found {len(image_paths)} images in {photos_dir}", flush=True)
    print(f"[+] Loading captioning model '{args.model_id}'...", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device.upper()}", flush=True)

    try:
        if _is_florence(args.model_id):
            processor, model, backend = _load_florence(args.model_id, device)
        else:
            processor, model, backend = _load_blip(args.model_id, device)
    except Exception as e:
        print(f"Error loading model '{args.model_id}': {e}", flush=True)
        sys.exit(1)

    print(f"[+] Backend: {backend}", flush=True)

    captions_dict: dict[str, str] = {}
    if captions_file.exists() and not args.force:
        try:
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            print(f"[+] Loaded {len(captions_dict)} existing captions from {captions_file}", flush=True)
        except Exception:
            captions_dict = {}

    if args.txt_out_dir:
        txt_out_dir = Path(args.txt_out_dir)
    else:
        parent_dir = photos_dir.parent
        if "kaggle/input" in str(parent_dir).lower() or "/input" in str(parent_dir).lower():
            txt_out_dir = Path("./captions")
            print(f"[!] Warning: photos_dir is read-only. Saving txt to '{txt_out_dir}'", flush=True)
        else:
            txt_out_dir = parent_dir / "captions"

    if args.save_txt:
        txt_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm

        pbar = tqdm(total=len(image_paths), desc="Captioning")
    except ImportError:
        pbar = None

    batch_paths: list[Path] = []
    batch_images: list[Image.Image] = []
    skipped = 0

    def should_skip(path: Path) -> bool:
        """Skip images that already have a good caption (unless --force)."""
        if args.force:
            return False
        existing = captions_dict.get(path.name) or captions_dict.get(path.stem)
        if not existing:
            return False
        if is_degenerate_caption(existing):
            return False
        # Keep good existing captions (default and --fix_bad_only).
        return True

    def process_batch(paths: list[Path], images: list[Image.Image]) -> None:
        try:
            if backend == "florence":
                raw_caps = _caption_florence(
                    processor, model, images, device, args.max_new_tokens
                )
            else:
                raw_caps = _caption_blip(
                    processor, model, images, device, args.max_new_tokens
                )

            for path, cap in zip(paths, raw_caps):
                clean_cap = normalize_product_caption(cap)
                captions_dict[path.name] = clean_cap
                captions_dict[path.stem] = clean_cap
                if args.save_txt:
                    with open(txt_out_dir / f"{path.stem}.txt", "w", encoding="utf-8") as tf:
                        tf.write(clean_cap)
        except Exception as e:
            print(f"Error processing batch: {e}", flush=True)

    for idx, img_path in enumerate(image_paths):
        if should_skip(img_path):
            skipped += 1
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

    print(f"\n[DONE] Saved {len(captions_dict)} captions → '{captions_file}'", flush=True)
    print(f"[DONE] Skipped (kept existing): {skipped}", flush=True)
    if args.save_txt:
        print(f"[DONE] Also wrote .txt files → '{txt_out_dir}'", flush=True)


if __name__ == "__main__":
    main()
