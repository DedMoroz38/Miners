"""Per-grain intergrowth type + image-level ore-sort decision (explicit rule)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import CLASS_BACKGROUND, CLASS_SULFIDE, CLASS_TALC
from .grains import Grain


@dataclass(frozen=True)
class RuleThresholds:
    talc_frac: float = 0.10
    fine_replacement: float = 0.40
    fine_max_area_um2: float = 5000.0
    fine_min_fragments: int = 4
    fine_dominance: float = 0.50


def grain_type(g: Grain, t: RuleThresholds = RuleThresholds()) -> str:
    """'fine' (тонкое) or 'normal' (обычное) intergrowth."""
    if g.replacement_ratio > t.fine_replacement:
        return "fine"
    if g.area_um2 < t.fine_max_area_um2 and g.fragmentation >= t.fine_min_fragments:
        return "fine"
    return "normal"


@dataclass(frozen=True)
class ImageResult:
    ore_class: str  # 'talc' | 'fine' | 'normal'
    talc_frac: float
    fine_frac: float  # share of sulfide-grain area that is 'fine'
    sulfide_frac: float
    grain_types: dict[int, str]


def classify_image(
    class_map: np.ndarray,
    grains: list[Grain],
    t: RuleThresholds = RuleThresholds(),
) -> ImageResult:
    valid = class_map != CLASS_BACKGROUND
    denom = max(int(valid.sum()), 1)
    talc_frac = float((class_map == CLASS_TALC).sum()) / denom
    sulfide_frac = float((class_map == CLASS_SULFIDE).sum()) / denom

    types = {g.label: grain_type(g, t) for g in grains}
    total_area = sum(g.area_um2 for g in grains)
    fine_area = sum(g.area_um2 for g in grains if types[g.label] == "fine")
    fine_frac = float(fine_area / total_area) if total_area > 0 else 0.0

    if talc_frac > t.talc_frac:
        ore = "talc"
    elif fine_frac > t.fine_dominance:
        ore = "fine"
    else:
        ore = "normal"
    return ImageResult(ore, talc_frac, fine_frac, sulfide_frac, types)
