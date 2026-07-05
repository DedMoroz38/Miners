"""Bake the canonical talc normalization into a copy of the YOLO-seg dataset.

Variant B, step 1. Reads the hand-labelled raw dataset
(ML/first_labling_attempt/yolo_seg) and writes an anchor-normalized twin
(dataset_norm/ by default), so train.py fine-tunes on exactly the profile the
panorama tiler will feed at inference (both call normalize.preprocess_talc).

What it does per image:
  * read with EXIF/truncation handling identical to the inference reader (PIL,
    LOAD_TRUNCATED_IMAGES) so train and inference see the same pixels;
  * apply normalize.preprocess_talc PER IMAGE (each field gets its own anchors —
    the labelled tiles ARE single fields, which is the granularity inference
    reproduces per fragment);
  * write the normalized image (PNG = lossless, no double-JPEG artefacts);
  * copy the polygon label file verbatim (normalization is pixel-wise; geometry
    is unchanged, so labels stay valid).

The train/val split is preserved exactly (we copy per split, never reshuffle),
so a normalized run is directly comparable to the raw baseline.

Usage:
    python build_dataset.py                       # -> ./dataset_norm
    python build_dataset.py --out /abs/other_dir
    python build_dataset.py --src /abs/yolo_seg   # different source dataset
    python build_dataset.py --no-denoise          # anchor only, skip median blur
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFile

from normalize import preprocess_talc

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some microscope JPEGs are truncated

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = (HERE.parent / "first_labling_attempt" / "yolo_seg").resolve()
SPLITS = ("train", "val")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = {0: "talc"}


def _read_bgr(path: Path) -> np.ndarray:
    """Read an image the same way inference does: PIL RGB -> BGR uint8."""
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build(src: Path, out: Path, denoise: bool) -> None:
    if not (src / "images" / "train").is_dir():
        raise SystemExit(
            f"Source dataset not found at {src}.\n"
            "Regenerate it first: cd ../first_labling_attempt && "
            "python export_to_yoloseg.py --from-api"
        )
    if out.exists():
        shutil.rmtree(out)

    total = 0
    for split in SPLITS:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        out_img = out / "images" / split
        out_lbl = out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        if not img_dir.is_dir():
            print(f"[build] warning: no {split} split at {img_dir}; skipping")
            continue
        n = 0
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            norm = preprocess_talc(_read_bgr(p), denoise=denoise)
            # lossless PNG keeps the normalized pixels exact (no re-JPEG drift)
            cv2.imwrite(str(out_img / f"{p.stem}.png"), norm)
            label = lbl_dir / f"{p.stem}.txt"
            if label.is_file():
                shutil.copy2(label, out_lbl / label.name)
            n += 1
        total += n
        print(f"[build] {split}: {n} images normalized -> {out_img}")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    (out / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[build] done: {total} images | denoise={denoise} | dataset -> {out}")
    print(f"[build] train with: python train.py --data {out / 'data.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="raw YOLO-seg dataset (images/{train,val}, labels/{train,val})")
    ap.add_argument("--out", default=str(HERE / "dataset_norm"),
                    help="destination for the normalized dataset")
    ap.add_argument("--no-denoise", action="store_true",
                    help="apply the exposure anchor only (skip the median blur)")
    args = ap.parse_args()
    build(Path(args.src), Path(args.out), denoise=not args.no_denoise)


if __name__ == "__main__":
    main()
