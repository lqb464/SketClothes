"""
Convert photos in data/png/photos to sketches in data/png/sketches.

Usage:
  python training/utils/photos_to_sketches.py
  python training/utils/photos_to_sketches.py --photos_dir path/to/photos --sketches_dir path/to/sketches
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


def smooth_contour(contour, window_size=5):
    """Smooth out pixel coordinate steps using a moving average filter."""
    if len(contour) < window_size:
        return contour
    pts = contour.squeeze(1) # shape (N, 2)
    smoothed_pts = np.copy(pts)
    
    half_w = window_size // 2
    padded_pts = np.pad(pts, ((half_w, half_w), (0, 0)), mode='wrap')
    
    # Simple moving average for coordinate path smoothing
    for i in range(2):
        smoothed_pts[:, i] = np.convolve(
            padded_pts[:, i], 
            np.ones(window_size) / window_size, 
            mode='valid'
        )
        
    return smoothed_pts.reshape(-1, 1, 2).astype(np.int32)


def canvas_to_control_sketch(
    image: Image.Image,
    size: int = 512,
    *,
    method: str = "clean_edge",
    thickness: float = 1.25,
    min_contour_area: float = 12.0,
    bilateral_d: int = 9,
    median_k: int = 7,
) -> Image.Image:
    """
    Match training sketch style (edge-like, white background, dark lines).

    Methods:
    - 'clean_edge' (Recommended): Bilateral filtering + Hybrid Canny/Adaptive edge + Morphological closing + noise removal.
      Preserves complete, continuous structural outlines while filtering out noisy fabric speckles.
    - 'adaptive': Adaptive Gaussian thresholding.
    - 'canny': Standard Canny edge detection.
    """
    # Calculate draw size and drawing thickness to support fractional thickness (e.g. 1.25)
    target_thickness = float(thickness)
    if target_thickness.is_integer():
        draw_scale = 1.0
        draw_thickness = int(target_thickness)
    else:
        # Scale drawing canvas, draw with integer, and downsample back
        draw_thickness = int(np.ceil(target_thickness * 1.5))
        draw_scale = draw_thickness / target_thickness

    draw_size = int(size * draw_scale)

    # 1. Resize to the drawing resolution using high-quality Lanczos resampling.
    # This prevents aliasing/jagged curves (răng cưa) and naturally filters out high-frequency noise.
    image_resized = image.resize((draw_size, draw_size), Image.Resampling.LANCZOS)
    rgb = np.array(image_resized.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if method == "clean_edge":
        # 1. Silhouette extraction (outer boundary of the garment/shoe)
        # Background is white (255), so threshold at 250 to get the foreground binary mask
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Clean up mask using morphological operations scaled for draw_scale
        clean_k = int(3 * draw_scale) // 2 * 2 + 1
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (clean_k, clean_k))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel_clean)

        # Get exact outer boundary line using Canny on the clean mask
        outer_edges = cv2.Canny(binary_clean, 50, 150)

        # 2. Coarse interior edges
        # Apply a strong median filter to completely dissolve high-frequency repeating textures (mesh, checkers, studs, text)
        med_k = int(median_k * draw_scale) // 2 * 2 + 1
        smoothed_gray = cv2.medianBlur(gray, med_k)
        
        # Use a scale-appropriate blur to completely smooth out fine lines and noise
        blur_k = int(5 * draw_scale) // 2 * 2 + 1
        blurred_coarse = cv2.GaussianBlur(smoothed_gray, (blur_k, blur_k), 0)
        interior_canny = cv2.Canny(blurred_coarse, 30, 80)

        # Avoid duplicate lines at the boundary by eroding the mask slightly
        erode_k = int(5 * draw_scale) // 2 * 2 + 1
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_k, erode_k))
        eroded_mask = cv2.erode(binary_clean, kernel_erode)
        interior_edges = cv2.bitwise_and(interior_canny, eroded_mask)

        # 3. Combine outer silhouette and coarse interior edges
        combined_edges = cv2.bitwise_or(outer_edges, interior_edges)

        # 4. Contour filtering and sharp drawing
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(combined_edges)

        # Scale min_contour_area for drawing scale
        min_area_scaled = min_contour_area * draw_scale

        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)
            area = cv2.contourArea(cnt)
            if length >= min_area_scaled or area >= min_area_scaled:
                # Smooth the coordinate sequence to remove step-like pixel staircases
                smoothed_cnt = smooth_contour(cnt, window_size=5)
                # Draw sharp lines with the calculated integer thickness at scaled size
                cv2.drawContours(clean_mask, [smoothed_cnt], -1, 255, thickness=draw_thickness)

        control = 255 - clean_mask

    elif method == "canny":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        if thickness > 1:
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        control = 255 - edges

    else:
        # FashionSD-X paper uses adaptive thresholding on garment images.
        control = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        if thickness > 1:
            # For adaptive, black lines are 0 on 255 white background
            inv = 255 - control
            kernel = np.ones((thickness, thickness), np.uint8)
            inv = cv2.dilate(inv, kernel, iterations=1)
            control = 255 - inv

    control_rgb = cv2.cvtColor(control, cv2.COLOR_GRAY2RGB)
    if draw_scale != 1.0:
        # Resize back to target size using bilinear interpolation to achieve fractional thickness (e.g. 1.25px)
        control_rgb = cv2.resize(control_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(control_rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert garment photos to clean structural sketches.")
    parser.add_argument(
        "--photos_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "photos"),
        help="Path to photos directory."
    )
    parser.add_argument(
        "--sketches_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "png" / "sketches"),
        help="Path to sketches output directory."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="clean_edge",
        choices=["clean_edge", "adaptive", "canny"],
        help="Sketch extraction method ('clean_edge', 'adaptive', or 'canny')."
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
        help="Minimum contour length/area to keep when using clean_edge (default: 12.0, filters out tiny fabric speckles while preserving lines)."
    )
    parser.add_argument(
        "--bilateral_d",
        type=int,
        default=9,
        help="Diameter for bilateral filter smoothing in clean_edge method."
    )
    parser.add_argument(
        "--median_k",
        type=int,
        default=7,
        help="Kernel size for median blur texture removal in clean_edge method (default: 7. Higher value = stronger texture removal)."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target resolution size for output sketches."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    photos_dir = Path(args.photos_dir)
    sketches_dir = Path(args.sketches_dir)

    if not photos_dir.exists():
        print(f"Error: Photos directory '{photos_dir}' does not exist.", flush=True)
        sys.exit(1)

    sketches_dir.mkdir(parents=True, exist_ok=True)

    # Support png, jpg, jpeg extensions
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(photos_dir.glob(ext))

    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in {photos_dir}", flush=True)
        return

    print(f"[+] Found {len(image_paths)} images in {photos_dir}", flush=True)
    print(f"[+] Converting to sketches using method='{args.method}' (size: {args.size}px, thickness: {args.thickness}px)...", flush=True)

    # Try importing tqdm for progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(image_paths, desc="Converting")
    except ImportError:
        pbar = image_paths

    success_count = 0
    for idx, img_path in enumerate(pbar):
        try:
            with Image.open(img_path) as img:
                sketch = canvas_to_control_sketch(
                    img,
                    size=args.size,
                    method=args.method,
                    thickness=args.thickness,
                    min_contour_area=args.min_contour_area,
                    bilateral_d=args.bilateral_d,
                    median_k=args.median_k,
                )
                out_path = sketches_dir / f"{img_path.stem}.png"
                sketch.save(out_path, format="PNG")
                success_count += 1
                if not hasattr(pbar, "update") and (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(image_paths)} images...", flush=True)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", flush=True)

    print(f"\n[DONE] Completed! Successfully converted {success_count}/{len(image_paths)} images to '{sketches_dir}'", flush=True)


if __name__ == "__main__":
    main()
