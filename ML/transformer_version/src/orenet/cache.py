"""One-off cache of preprocessed images + label maps to disk.

GMM pseudo-labelling and blue-contour extraction are slow; we run them once and
store PNGs so training epochs just read files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .data_index import Record
from .labels import build_label
from .preprocess import PreprocessConfig


@dataclass(frozen=True)
class CachedItem:
    image_path: Path
    label_path: Path
    sort: str
    has_talc: bool
    slide_id: str


def build_cache(
    records: list[Record],
    cache_dir: Path,
    pre_cfg: PreprocessConfig = PreprocessConfig(),
    overwrite: bool = False,
) -> list[CachedItem]:
    """Materialise (image, label) PNG pairs; return the cached index."""
    img_dir = cache_dir / "images"
    lab_dir = cache_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    items: list[CachedItem] = []
    for i, rec in enumerate(tqdm(records, desc="caching")):
        stem = f"{i:05d}_{rec.slide_id}"
        ip = img_dir / f"{stem}.png"
        lp = lab_dir / f"{stem}.png"
        if overwrite or not (ip.exists() and lp.exists()):
            try:
                img, label = build_label(rec, pre_cfg)
            except Exception as exc:  # noqa: BLE001 - skip unreadable files, keep going
                print(f"skip {rec.image_path.name}: {exc}")
                continue
            cv2.imwrite(str(ip), img)
            cv2.imwrite(str(lp), label)
        items.append(
            CachedItem(ip, lp, rec.sort, rec.talc_label is not None, rec.slide_id)
        )
    return items
