"""Grain-level intergrowth analysis on the phase class map.

For each sulfide aggregate we build its convex hull ("the grain if it were not
replaced") and measure how much of that hull is occupied by non-sulfide phases.
High replacement -> тонкие срастания; low -> обычные срастания. This is a
transparent geometric formalisation of the geologist's definition
(cf. Pérez-Barnuevo et al., Minerals Engineering 52, 2013).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import binary_closing, disk

from .constants import CLASS_GRAY, CLASS_SULFIDE


@dataclass(frozen=True)
class GrainParams:
    aggregate_px: int = 15
    min_area_um2: float = 100.0


@dataclass(frozen=True)
class Grain:
    label: int
    area_um2: float
    replacement_ratio: float
    fragmentation: int
    boundary_complexity: float
    contact_with_gray: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


def extract_grains(
    class_map: np.ndarray,
    um_per_px: float = 1.0,
    params: GrainParams = GrainParams(),
) -> list[Grain]:
    sulf = class_map == CLASS_SULFIDE
    if sulf.sum() == 0:
        return []
    agg = ndi.binary_closing(sulf, disk(params.aggregate_px))
    labels, n = ndi.label(agg)
    frag_labels, _ = ndi.label(ndi.binary_closing(sulf, disk(3)))

    grains: list[Grain] = []
    for rp in regionprops(labels):
        comp = labels == rp.label
        sulf_in = comp & sulf
        area_um2 = float(sulf_in.sum()) * um_per_px**2
        if area_um2 < params.min_area_um2:
            continue
        hull = rp.image_convex
        hull_area = float(hull.sum())
        sulf_area = float(sulf_in.sum())
        replacement = 1.0 - sulf_area / max(hull_area, 1.0)

        frag = int(len(np.unique(frag_labels[sulf_in])) - (0 in np.unique(frag_labels[sulf_in])))
        perim = float(rp.perimeter)
        equiv = 2.0 * np.sqrt(np.pi * max(sulf_area, 1.0))
        complexity = perim / max(equiv, 1.0)

        border = comp ^ ndi.binary_erosion(comp)
        dil = ndi.binary_dilation(comp, iterations=2) & ~comp
        gray_touch = (dil & (class_map == CLASS_GRAY)).sum()
        contact_gray = float(gray_touch) / max(border.sum(), 1)

        grains.append(
            Grain(
                label=rp.label,
                area_um2=area_um2,
                replacement_ratio=float(np.clip(replacement, 0, 1)),
                fragmentation=max(frag, 1),
                boundary_complexity=complexity,
                contact_with_gray=float(np.clip(contact_gray, 0, 1)),
                centroid=(float(rp.centroid[0]), float(rp.centroid[1])),
                bbox=tuple(int(v) for v in rp.bbox),
            )
        )
    return grains
