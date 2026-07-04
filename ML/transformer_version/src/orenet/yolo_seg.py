"""Talc supervision from the hand-labelled YOLO-seg dataset.

Replaces the old blue-contour parser. The dataset (produced by
`first_labling_attempt/export_to_yoloseg.py`) stores, per image, one polygon per
line in `labels/{train,val}/<stem>.txt`:

    0 x1 y1 x2 y2 ... xn yn      # class 0 = talc, coords normalised to [0,1]

`filename_map.json` maps each sanitised stem back to the original file name, so
we can attach a talc label to the corresponding original image in `data/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _label_for_image(yolo_dir: Path, split: str, stem: str) -> Path:
    return yolo_dir / "labels" / split / f"{stem}.txt"


def build_talc_lookup(yolo_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map original image *basename* -> (label .txt path, split 'train'|'val').

    Uses filename_map.json (sanitised stem -> original path) to key by the
    original file name that also lives under data/.
    """
    fmap_path = yolo_dir / "filename_map.json"
    if not fmap_path.exists():
        return {}
    fmap: dict[str, str] = json.loads(fmap_path.read_text(encoding="utf-8"))

    lookup: dict[str, tuple[Path, str]] = {}
    for out_name, original in fmap.items():
        stem = Path(out_name).stem
        original_base = Path(original).name  # e.g. "2550374-2 10х.JPG"
        for split in ("train", "val"):
            lbl = _label_for_image(yolo_dir, split, stem)
            if lbl.exists():
                lookup[original_base] = (lbl, split)
                break
    return lookup


def rasterize_talc(label_path: Path, height: int, width: int) -> np.ndarray:
    """Fill YOLO-seg talc polygons into a uint8 {0,1} mask of the given size."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + >=3 points
            continue
        coords = np.array(parts[1:], dtype=np.float32)
        pts = coords.reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], color=1)
    return mask
