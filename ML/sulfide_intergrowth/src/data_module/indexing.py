"""Dataset indexing: scan class folders, derive slide groups, holdout split.

Labels are IMAGE-LEVEL (from folder names): 0 = normal intergrowths
("рядовые"), 1 = fine intergrowths ("тонкие" / "труднообогатимые").
Talc folders are ignored entirely by design.

Slide grouping (leakage control):
- part1 names like "2539439-3.JPG" carry a real polished-section id -> group
  by the leading 6-8 digit id;
- part2 names are bare frame counters ("104.JPG", "104_.jpg", "106 (2).JPG"):
  we normalize away duplicate suffixes so obvious re-shots share a group, but
  true slide identity is unknown there. The final metric therefore also uses a
  holdout that prefers strongly-grouped part1 slides.
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_NAMES = {0: "normal", 1: "fine"}

_SLIDE_ID_RE = re.compile(r"^(\d{6,8})")
_DUP_SUFFIX_RE = re.compile(r"[\s_\-]*(\(\d+\))?[\s_\-]*$")


def slide_group(path: Path) -> str:
    """Derive a leakage-safe group key for an image path."""
    stem = path.stem
    m = _SLIDE_ID_RE.match(stem)
    if m:
        return f"slide_{m.group(1)}"
    norm = _DUP_SUFFIX_RE.sub("", stem).lower()
    return f"{path.parent.name}/{norm or stem.lower()}"


def _class_of(folder_name: str, class_dirs: dict[str, list[str]],
              ignore_dirs: list[str]) -> int | None:
    low = folder_name.lower()
    if any(pat.lower() in low for pat in ignore_dirs):
        return None
    for pat in class_dirs.get("fine", []):
        if pat.lower() in low:
            return 1
    for pat in class_dirs.get("normal", []):
        if pat.lower() in low:
            return 0
    return None


def build_index(data_root: Path, part_dirs: list[str],
                class_dirs: dict[str, list[str]], ignore_dirs: list[str],
                holdout_fraction: float, seed: int,
                limit_per_class: int | None = None) -> pd.DataFrame:
    """Scan data folders and build the image index with groups and splits.

    Returns:
        DataFrame with columns: path, label, label_name, group, part, split
        (split in {"trainval", "holdout"}).

    Raises:
        FileNotFoundError: when no labelled images are found.
    """
    rows: list[dict] = []
    for part in part_dirs:
        part_path = data_root / part
        if not part_path.is_dir():
            logger.warning("Part dir missing: %s", part_path)
            continue
        for sub in sorted(p for p in part_path.iterdir() if p.is_dir()):
            label = _class_of(sub.name, class_dirs, ignore_dirs)
            if label is None:
                logger.info("Ignoring folder (talc/unknown): %s", sub.name)
                continue
            for img in sorted(sub.rglob("*")):
                if img.suffix.lower() in IMG_EXTS and img.is_file():
                    rows.append({
                        "path": str(img),
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "group": slide_group(img),
                        "part": part,
                    })
    if not rows:
        raise FileNotFoundError(f"No labelled images under {data_root}")
    df = pd.DataFrame(rows)
    if limit_per_class is not None:
        parts = [g.sample(min(len(g), limit_per_class), random_state=seed)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).reset_index(drop=True)
        logger.info("limit_per_class=%d -> %d images", limit_per_class, len(df))
    df["split"] = _assign_holdout(df, holdout_fraction, seed)
    logger.info("Index: %d images, %d groups, labels %s, holdout %d",
                len(df), df["group"].nunique(),
                df["label"].value_counts().to_dict(),
                int((df["split"] == "holdout").sum()))
    return df


def _assign_holdout(df: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    """Group-aware stratified holdout, preferring strongly-grouped slides."""
    rng = np.random.default_rng(seed)
    split = pd.Series("trainval", index=df.index)
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        groups = sub.groupby("group").size()
        strong = [g for g in groups.index if g.startswith("slide_")]
        weak = [g for g in groups.index if not g.startswith("slide_")]
        rng.shuffle(strong)
        rng.shuffle(weak)
        target = int(round(len(sub) * fraction))
        picked, count = [], 0
        for g in strong + weak:
            if count >= target:
                break
            picked.append(g)
            count += int(groups[g])
        split[sub[sub["group"].isin(picked)].index] = "holdout"
    return split


def save_index(df: pd.DataFrame, path: Path) -> None:
    """Persist the index as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Index saved: %s", path)


def load_index(path: Path) -> pd.DataFrame:
    """Load a previously built index.

    Raises:
        FileNotFoundError: when the index CSV does not exist yet.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Index not found: {path}. Run build_index.py first.")
    return pd.read_csv(path)
