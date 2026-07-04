"""Method 1 — minerallurgical indices (reimplemented from the paper).

Pérez-Barnuevo, Pirard & Castroviejo (2013), "Automated characterisation of
intergrowth textures in mineral particles. A case study", Minerals Engineering
52:136-142.  https://doi.org/10.1016/j.mineng.2013.05.001

NOT the authors' original code (none was released) and NOT their proprietary
particle data. We follow the paper's recipe: quantify each mineral aggregate with
a set of intergrowth/replacement indices, then classify by discriminant analysis
(see lda_model.py). Because our supervision is image-level (folder sort), the
per-grain indices are aggregated into one feature vector per image.

Phase map (sulfide / gray-ore / matrix) comes from the orenet GMM pseudo-labeller
so this module is self-contained and needs no trained segmenter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .constants import ORENET_SRC


def _ensure_orenet() -> None:
    p = str(ORENET_SRC.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


# Names of the aggregated image-level features (order matches image_features()).
FEATURE_NAMES: list[str] = [
    "area_wt_replacement",     # area-weighted mean replacement ratio
    "fine_area_fraction",      # share of grain area with replacement > 0.4
    "mean_boundary_complex",   # perimeter / equivalent-circle perimeter
    "mean_fragmentation",      # sulfide fragments per aggregate
    "mean_contact_gray",       # fraction of grain border touching gray ore
    "grain_density",           # grains per megapixel of valid area
    "median_grain_area_um2",
    "area_cv",                 # coefficient of variation of grain areas
    "sulfide_area_fraction",
]


def image_features(image_path: Path, fine_replacement: float = 0.40) -> np.ndarray:
    """Return the PB2013-style intergrowth feature vector for one image."""
    _ensure_orenet()
    from orenet.constants import CLASS_BACKGROUND, CLASS_SULFIDE
    from orenet.grains import extract_grains
    from orenet.preprocess import PreprocessConfig, preprocess
    from orenet.pseudolabels import pseudo_label

    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(image_path)
    img = preprocess(raw, PreprocessConfig())
    phases = pseudo_label(img)
    grains = extract_grains(phases)

    valid = int((phases != CLASS_BACKGROUND).sum())
    sulfide_frac = float((phases == CLASS_SULFIDE).sum()) / max(valid, 1)

    if not grains:
        # no sulfide aggregates: everything zero except (zero) sulfide fraction
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, sulfide_frac], dtype=np.float32)

    areas = np.array([g.area_um2 for g in grains], dtype=np.float64)
    repl = np.array([g.replacement_ratio for g in grains], dtype=np.float64)
    total_area = areas.sum()

    area_wt_repl = float((repl * areas).sum() / max(total_area, 1e-6))
    fine_area = float(areas[repl > fine_replacement].sum())
    fine_area_frac = fine_area / max(total_area, 1e-6)
    mean_complex = float(np.mean([g.boundary_complexity for g in grains]))
    mean_frag = float(np.mean([g.fragmentation for g in grains]))
    mean_contact = float(np.mean([g.contact_with_gray for g in grains]))
    density = len(grains) / max(valid / 1e6, 1e-6)
    median_area = float(np.median(areas))
    area_cv = float(np.std(areas) / max(np.mean(areas), 1e-6))

    return np.array(
        [area_wt_repl, fine_area_frac, mean_complex, mean_frag, mean_contact,
         density, median_area, area_cv, sulfide_frac],
        dtype=np.float32,
    )
