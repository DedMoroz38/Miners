"""Sample source with optional reuse of the transformer_version preprocessed cache.

The heavy cost in the segmenter notebook was building the preprocessed/denoised
PNGs (Lab illumination fix + CLAHE). We reuse those exact PNGs as classifier input
when present, mapping each cached image back to its ore-sort via slide id. Falls
back to the raw JPG when no cache is found, so it runs anywhere.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# reuse the intergrowth data indexing (folder -> label, slide-grouped split)
_ISRC = Path(__file__).resolve().parents[3] / "intergrowth" / "src"
if str(_ISRC) not in sys.path:
    sys.path.insert(0, str(_ISRC))

from intergrowth.data import Sample, build_samples, group_split  # noqa: E402
from intergrowth.constants import NORMAL, FINE  # noqa: E402

# where the transformer_version notebook writes its preprocessed cache
CACHE_DIRS = (
    Path("../transformer_version/data/derived/cache/train/images"),
    Path("../transformer_version/data/derived/cache/eval/images"),
    Path("../transformer_version/data/derived/subset/train/images"),
    Path("../transformer_version/data/derived/subset/eval/images"),
)


@dataclass(frozen=True)
class Item:
    path: Path        # preprocessed PNG if cached, else raw image
    label: int
    slide_id: str
    cached: bool


def _cache_lookup() -> dict[str, Path]:
    """slide_id -> a cached preprocessed PNG (filenames are '{i:05d}_{slide}.png')."""
    lut: dict[str, Path] = {}
    for d in CACHE_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            slide = p.stem.split("_", 1)[-1]
            lut.setdefault(slide, p)
    return lut


def build_items(prefer_cache: bool = True) -> list[Item]:
    """Non-talc images as Items, reusing cached preprocessed PNGs when available."""
    samples = build_samples()
    lut = _cache_lookup() if prefer_cache else {}
    items: list[Item] = []
    for s in samples:
        cp = lut.get(s.slide_id)
        if cp is not None:
            items.append(Item(cp, s.label, s.slide_id, True))
        else:
            items.append(Item(s.image_path, s.label, s.slide_id, False))
    return items


def split_items(items: list[Item], test_size: float = 0.2, seed: int = 42):
    """Slide-grouped split (no leakage), mirroring intergrowth.group_split."""
    proxy = [Sample(it.path, it.label, it.slide_id) for it in items]
    tr, te = group_split(proxy, test_size=test_size, seed=seed)
    tr_ids = {s.slide_id for s in tr}
    return ([it for it in items if it.slide_id in tr_ids],
            [it for it in items if it.slide_id not in tr_ids])


__all__ = ["Item", "build_items", "split_items", "NORMAL", "FINE"]
