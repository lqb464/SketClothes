"""
Chuyển dataset Parquet chứa ảnh.

MODE:
    rgba2parquet
        transparent*.parquet
            ↓
        white-background*.parquet

    parquet2png
        white-background*.parquet
            ↓
        data/png/photos/*.png

    all
        chạy cả hai bước trên liên tiếp
"""

import io
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

from pathlib import Path

RUN_MODE = "all"
# "rgba2parquet"
# "parquet2png"
# "all"

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "data" / "parquet"

TRANSPARENT_DIR = PARQUET_DIR / "transparent"
WHITE_BG_DIR = PARQUET_DIR / "white-background"

PNG_DIR = ROOT / "data" / "png" / "photos"

IMAGE_COL = "image"


# ============================================================
# IMAGE
# ============================================================

def rgba_to_white_bg(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])

    return bg.convert("RGB")


def get_image_bytes(cell):
    if isinstance(cell, dict):
        return cell["bytes"]
    return bytes(cell)


# ============================================================
# STEP 1
# transparent.parquet
#      ↓
# white-background.parquet
# ============================================================


def convert_transparent_to_white(input_file: Path):

    print(f"\nĐọc: {input_file.name}")

    df = pd.read_parquet(input_file)

    converted = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=input_file.stem,
    ):

        raw = get_image_bytes(row[IMAGE_COL])

        img = rgba_to_white_bg(raw)

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        converted.append(buf.getvalue())

    df[IMAGE_COL] = converted

    WHITE_BG_DIR.mkdir(parents=True, exist_ok=True)

    output_name = input_file.name.replace(
        "transparent",
        "white-background",
        1,  # chỉ thay lần xuất hiện đầu tiên
    )

    output_path = WHITE_BG_DIR / output_name

    df.to_parquet(output_path, index=False)

    print(f"✅ Saved -> {output_path}")


# ============================================================
# STEP 2
# parquet
#    ↓
# png
# ============================================================

def parquet_to_png(input_file: Path):

    print(f"\nXuất PNG: {input_file.name}")

    df = pd.read_parquet(input_file)

    out_dir = PNG_DIR / input_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=input_file.stem,
    ):

        raw = get_image_bytes(row[IMAGE_COL])

        img = Image.open(io.BytesIO(raw)).convert("RGB")

        img.save(out_dir / f"{idx:06d}.png")

    print(f"✅ PNG -> {out_dir}")


# ============================================================
# MAIN
# ============================================================

def run_rgba2parquet():

    files = sorted(TRANSPARENT_DIR.glob("*.parquet"))

    if not files:
        print("Không tìm thấy parquet trong thư mục transparent")
        return

    print(f"Tìm thấy {len(files)} parquet")

    for f in files:
        convert_transparent_to_white(f)


def run_parquet2png():

    files = sorted(WHITE_BG_DIR.glob("*.parquet"))

    if not files:
        print("Không tìm thấy parquet trong white-background")
        return

    PNG_DIR.mkdir(parents=True, exist_ok=True)

    counter = 0

    for parquet_file in files:

        print(f"\nXuất PNG: {parquet_file.name}")

        df = pd.read_parquet(parquet_file)

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=parquet_file.stem,
        ):

            raw = get_image_bytes(row[IMAGE_COL])

            img = Image.open(io.BytesIO(raw)).convert("RGB")

            img.save(PNG_DIR / f"{counter:06d}.png")

            counter += 1

    print(f"\n✅ Đã xuất {counter} ảnh vào {PNG_DIR}")


def main():

    if RUN_MODE == "rgba2parquet":

        run_rgba2parquet()

    elif RUN_MODE == "parquet2png":

        run_parquet2png()

    elif RUN_MODE == "all":

        run_rgba2parquet()
        run_parquet2png()

    else:

        raise ValueError(
            "RUN_MODE phải là: rgba2parquet | parquet2png | all"
        )


if __name__ == "__main__":
    main()