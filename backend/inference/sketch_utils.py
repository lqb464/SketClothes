import base64
import io

import cv2
import numpy as np
from PIL import Image

import config


def decode_sketch_b64(sketch_b64: str) -> Image.Image:
    if "," in sketch_b64:
        sketch_b64 = sketch_b64.split(",", 1)[1]
    raw = base64.b64decode(sketch_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return preprocess_sketch(image)


def canvas_to_control_sketch(image: Image.Image, size: int = 512) -> Image.Image:
    """Convert paint-style canvas to edge sketch (matches training conditioning)."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if config.SKETCH_PREPROCESS == "canny":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        control = 255 - edges
    elif config.SKETCH_PREPROCESS == "none":
        return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    else:
        control = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    control_rgb = cv2.cvtColor(control, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(control_rgb).resize((size, size), Image.Resampling.LANCZOS)


def preprocess_sketch(image: Image.Image, size: int | None = None) -> Image.Image:
    """Resize and convert sketch for ControlNet conditioning."""
    size = size or config.RESOLUTION
    if config.SKETCH_PREPROCESS == "none":
        return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return canvas_to_control_sketch(image, size=size)


def is_blank_sketch(image: Image.Image, min_colored_pixels: int = 8) -> bool:
    """Return True if sketch has too few non-white pixels."""
    pixels = image.load()
    width, height = image.size
    colored = 0
    step = max(1, int((width * height / 4096) ** 0.5))

    for y in range(0, height, step):
        for x in range(0, width, step):
            r, g, b = pixels[x, y]
            if r + g + b < 750:
                colored += 1
                if colored >= min_colored_pixels:
                    return False
    return True


def image_to_jpeg_b64(image: Image.Image, quality: int = 85) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
