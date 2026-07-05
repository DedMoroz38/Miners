"""Canonical photometric normalization for the talc segmenter (variant B).

UNIFIED with the sulfide pipeline: this module does NOT re-implement the
transform — it loads and calls the sulfide pipeline's own
`preprocess_image` (ML/sulfide_intergrowth/src/data_module/preprocessing.py).
So the talc segmenter is retrained in the EXACT SAME colour profile the sulfide
models are trained/inferred in — one implementation, zero drift.

`PreprocessConfig()` defaults (illumination="anchor", downscale=8, sigma=65,
denoise="median", denoise_h=5, clahe=0) are identical to the sulfide inference
config `run/conf/data/default.yaml::preprocess`, so `preprocess_talc(bgr)` ==
what the sulfide segmenter sees at inference.

Applied at BOTH phases: build_dataset.py bakes it into the training tiles, and
the panorama tiler in ML/talc_infer/ imports `preprocess_talc` and applies it
PER FRAGMENT. Deps: numpy + opencv only (the sulfide preprocessing file has no
hydra/relative imports, so we load it standalone — no sulfide venv needed).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Single source of truth: the sulfide pipeline's preprocessing module.
_SULFIDE_PREPROCESS = (
    Path(__file__).resolve().parent.parent
    / "sulfide_intergrowth" / "src" / "data_module" / "preprocessing.py"
)


def _load_sulfide_preprocessing():
    """Import the sulfide preprocessing FILE directly (bypasses the package
    __init__, which would drag in hydra-dependent siblings)."""
    if not _SULFIDE_PREPROCESS.is_file():
        raise FileNotFoundError(
            f"Sulfide preprocessing not found at {_SULFIDE_PREPROCESS}. "
            "The talc profile is unified with the sulfide pipeline — that file "
            "is the shared source of truth and must be present."
        )
    spec = importlib.util.spec_from_file_location(
        "sulfide_preprocessing", _SULFIDE_PREPROCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pp = _load_sulfide_preprocessing()

# Re-exported so callers/tests can prove they use the sulfide implementation.
PreprocessConfig = _pp.PreprocessConfig
preprocess_image = _pp.preprocess_image
normalize_exposure_anchor = _pp.normalize_exposure_anchor

# The canonical profile == sulfide inference defaults (anchor + median denoise).
_TALC_CFG = PreprocessConfig()                       # denoise="median"
_TALC_CFG_NO_DENOISE = PreprocessConfig(denoise="none")


def preprocess_talc(bgr: np.ndarray, denoise: bool = True) -> np.ndarray:
    """Bring one tile/fragment to the shared sulfide colour profile.

    Thin wrapper over the sulfide `preprocess_image` so train and inference use
    the identical transform. Applied PER TILE / PER FRAGMENT (anchors are
    computed from the image passed in) — pass one field-scale crop at a time,
    never the whole panorama.

    Args:
        bgr: uint8 BGR image (single micrograph tile or one panorama fragment).
        denoise: keep the sulfide default median(3); False = anchor only.

    Returns:
        uint8 BGR image of the same shape, in the canonical (sulfide) profile.
    """
    return preprocess_image(bgr, _TALC_CFG if denoise else _TALC_CFG_NO_DENOISE)
