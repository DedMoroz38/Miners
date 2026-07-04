"""Scan the dataset folders and build a record table.

A record = one image with: path, ore-sort label (from folder), an optional talc
label (hand-labelled YOLO-seg polygons), and a slide id (for grouped splits).
Talc supervision comes from the YOLO-seg export via `yolo_seg.build_talc_lookup`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .constants import IMAGE_EXTS, PART1, PART2, SORT_FOLDERS, YOLO_SEG_DIR
from .yolo_seg import build_talc_lookup


@dataclass(frozen=True)
class Record:
    image_path: Path
    sort: str  # "normal" | "fine" | "talc"
    talc_label: Path | None  # YOLO-seg .txt with talc polygons, if hand-labelled
    talc_split: str | None  # "train" | "val" from the YOLO-seg export, if any
    slide_id: str


def _slide_id(name: str) -> str:
    """Group id for split: leading catalogue number or DSCN id, else full stem."""
    m = re.match(r"(\d{5,})", name)
    if m:
        return m.group(1)
    m = re.match(r"(DSCN\d+)", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return Path(name).stem


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def build_index(yolo_seg_dir: Path = YOLO_SEG_DIR) -> list[Record]:
    """Walk part1 + part2, attaching hand talc labels by original file name."""
    talc_lookup = build_talc_lookup(yolo_seg_dir)  # basename -> (label.txt, split)

    records: list[Record] = []
    for base in (PART1, PART2):
        if not base.exists():
            continue
        for folder in base.iterdir():
            if not folder.is_dir() or folder.name not in SORT_FOLDERS:
                continue
            sort = SORT_FOLDERS[folder.name]
            for img in sorted(folder.iterdir()):
                if not _is_image(img):
                    continue
                hit = talc_lookup.get(img.name)
                label_path, split = hit if hit else (None, None)
                records.append(
                    Record(
                        image_path=img,
                        sort=sort,
                        talc_label=label_path,
                        talc_split=split,
                        slide_id=_slide_id(img.name),
                    )
                )
    return records


def talc_records(records: list[Record]) -> list[Record]:
    return [r for r in records if r.talc_label is not None]
