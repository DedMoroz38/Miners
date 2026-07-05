"""Index the ore photos by folder label and split by slide (no leakage).

Only NON-talc folders are indexed (see constants.FOLDER_TO_LABEL). The image-level
label is the geologist's folder sort: рядовая (NORMAL) vs труднообогатимая (FINE).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from .constants import DATA_ROOTS, FOLDER_TO_LABEL, IMAGE_EXTS


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int      # NORMAL | FINE
    slide_id: str   # group key for a leak-free split


def _slide_id(name: str) -> str:
    """Leading catalogue number or DSCN id groups crops of the same slide."""
    m = re.match(r"(\d{5,})", name)
    if m:
        return m.group(1)
    m = re.match(r"(DSCN\d+)", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return Path(name).stem


def build_samples(data_roots: tuple[Path, ...] = DATA_ROOTS) -> list[Sample]:
    """Walk the dataset folders and collect labelled non-talc images."""
    samples: list[Sample] = []
    for root in data_roots:
        if not root.exists():
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name not in FOLDER_TO_LABEL:
                continue
            label = FOLDER_TO_LABEL[folder.name]
            for img in sorted(folder.iterdir()):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    samples.append(Sample(img, label, _slide_id(img.name)))
    return samples


def group_split(
    samples: list[Sample], test_size: float = 0.2, seed: int = 42
) -> tuple[list[Sample], list[Sample]]:
    """Split into (train, test) so no slide appears in both."""
    if len(samples) < 5:
        return samples, samples
    groups = [s.slide_id for s in samples]
    tr, te = next(
        GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(samples, groups=groups)
    )
    return [samples[i] for i in tr], [samples[i] for i in te]
