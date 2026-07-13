import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

# Import sketch extraction function from photos_to_sketches
try:
    from photos_to_sketches import canvas_to_control_sketch
except ImportError:
    # Handle path when executed directly
    sys.path.append(str(Path(__file__).resolve().parent))
    from photos_to_sketches import canvas_to_control_sketch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment dataset by horizontal flipping, zoom, random rotation, translation, brightness, and contrast reduction.")
    parser.add_argument(
        "--mode",
        type=str,
        default="flip",
        choices=["flip", "zoom", "rotate", "shift"],
        help="Augmentation mode ('flip' = horizontal flip + contrast; 'zoom' = zoom out + contrast; 'rotate' = zoom out + random rotation [-30, 30] + contrast; 'shift' = random translation [+-10%%] + random brightness [0.8, 1.2] + contrast)."
    )
    parser.add_argument(
        "--photos_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "photos"),
        help="Path to original photos directory."
    )
    parser.add_argument(
        "--out_photos_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "photos_augmented"),
        help="Path to save augmented photos."
    )
    parser.add_argument(
        "--out_sketches_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "sketches_augmented"),
        help="Path to save augmented sketches."
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "captions.json"),
        help="Path to original captions.json."
    )
    parser.add_argument(
        "--out_captions_file",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "captions_augmented.json"),
        help="Path to save augmented captions JSON."
    )
    parser.add_argument(
        "--contrast_factor",
        type=float,
        default=0.7,
        help="Contrast enhancement factor (default: 0.7. Values < 1.0 reduce contrast)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="clean_edge",
        choices=["clean_edge", "adaptive", "canny"],
        help="Sketch extraction method (default: clean_edge)."
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=1.25,
        help="Line thickness in pixels (default: 1.25)."
    )
    parser.add_argument(
        "--min_contour_area",
        type=float,
        default=12.0,
        help="Minimum contour length/area to keep (default: 12.0)."
    )
    parser.add_argument(
        "--median_k",
        type=int,
        default=7,
        help="Kernel size for median blur texture removal (default: 7)."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target resolution size for output sketches (default: 512)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    photos_dir = Path(args.photos_dir)
    
    # Dynamically determine output directories based on mode to avoid mixing
    suffix = f"_{args.mode}" if args.mode in ("zoom", "rotate", "shift") else ""
    out_photos_dir = Path(args.out_photos_dir + suffix)
    out_sketches_dir = Path(args.out_sketches_dir + suffix)
    captions_file = Path(args.captions_file)
    out_captions_file = Path(str(args.out_captions_file).replace(".json", f"{suffix}.json"))

    if not photos_dir.exists():
        print(f"Error: Original photos directory '{photos_dir}' does not exist.", flush=True)
        sys.exit(1)

    # Create target directories
    out_photos_dir.mkdir(parents=True, exist_ok=True)
    out_sketches_dir.mkdir(parents=True, exist_ok=True)

    # Find original images
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(photos_dir.glob(ext))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No original images found in {photos_dir}", flush=True)
        sys.exit(1)

    print(f"[+] Found {len(image_paths)} original images.", flush=True)
    print(f"[+] Augmenting: Flipping Left-Right + Reducing Contrast (factor: {args.contrast_factor})...", flush=True)
    print(f"[+] Generating sketches using method='{args.method}' (size: {args.size}px, thickness: {args.thickness}px)...", flush=True)

    try:
        from tqdm import tqdm
        pbar = tqdm(image_paths, desc="Augmenting")
    except ImportError:
        pbar = image_paths

    success_count = 0
    for idx, img_path in enumerate(pbar):
        try:
            with Image.open(img_path) as img:
                if args.mode == "zoom":
                    # Zoom out 10%: Scale to 90% and pad with white background
                    w, h = img.size
                    new_w = int(w * 0.9)
                    new_h = int(h * 0.9)
                    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    flipped_img = Image.new("RGB", (w, h), (255, 255, 255))
                    paste_x = (w - new_w) // 2
                    paste_y = (h - new_h) // 2
                    flipped_img.paste(scaled_img, (paste_x, paste_y))
                elif args.mode == "rotate":
                    # Zoom out 10% + Random Rotation [-30, 30] with white padding
                    import random
                    w, h = img.size
                    new_w = int(w * 0.9)
                    new_h = int(h * 0.9)
                    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    angle = random.uniform(-30.0, 30.0)
                    rotated_scaled = scaled_img.rotate(
                        angle, 
                        resample=Image.Resampling.BICUBIC, 
                        expand=False, 
                        fillcolor=(255, 255, 255)
                    )
                    
                    flipped_img = Image.new("RGB", (w, h), (255, 255, 255))
                    paste_x = (w - new_w) // 2
                    paste_y = (h - new_h) // 2
                    flipped_img.paste(rotated_scaled, (paste_x, paste_y))
                elif args.mode == "shift":
                    # Random Translation (Shift up to +-10%) + Random Brightness [0.8, 1.2] with white padding
                    import random
                    w, h = img.size
                    dx = int(random.uniform(-0.1, 0.1) * w)
                    dy = int(random.uniform(-0.1, 0.1) * h)
                    
                    # Shift image on white background
                    shifted = Image.new("RGB", (w, h), (255, 255, 255))
                    shifted.paste(img, (dx, dy))
                    
                    # Random Brightness Jitter
                    bright_factor = random.uniform(0.8, 1.2)
                    flipped_img = ImageEnhance.Brightness(shifted).enhance(bright_factor)
                else:
                    # Flip Left-Right
                    flipped_img = ImageOps.mirror(img)

                # 2. Reduce Contrast
                enhancer = ImageEnhance.Contrast(flipped_img)
                aug_img = enhancer.enhance(args.contrast_factor)

                # Save augmented photo (retaining the same filename to align them)
                out_img_path = out_photos_dir / img_path.name
                aug_img.save(out_img_path)

                # 3. Generate and save sketch from augmented photo
                sketch = canvas_to_control_sketch(
                    aug_img,
                    size=args.size,
                    method=args.method,
                    thickness=args.thickness,
                    min_contour_area=args.min_contour_area,
                    median_k=args.median_k
                )
                out_sketch_path = out_sketches_dir / f"{img_path.stem}.png"
                sketch.save(out_sketch_path, format="PNG")
                
                success_count += 1
                if not hasattr(pbar, "update") and (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(image_paths)} images...", flush=True)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", flush=True)

    print(f"[DONE] Created {success_count} augmented photos in '{out_photos_dir}'", flush=True)
    print(f"[DONE] Generated {success_count} sketches in '{out_sketches_dir}'", flush=True)

    # 4. Handle Captions if captions.json is present
    if captions_file.exists():
        try:
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            
            aug_captions_dict = {}
            for k, v in captions_dict.items():
                # Map the same caption to the augmented photo filename
                aug_captions_dict[k] = v
            
            with open(out_captions_file, "w", encoding="utf-8") as f:
                json.dump(aug_captions_dict, f, indent=2, ensure_ascii=False)
            print(f"[DONE] Saved augmented captions mapping to '{out_captions_file}'", flush=True)
        except Exception as e:
            print(f"Warning: Could not process captions file: {e}", flush=True)


if __name__ == "__main__":
    main()
