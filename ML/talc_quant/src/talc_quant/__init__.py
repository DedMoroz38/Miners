"""Talc quantification: 5-class segmentation + calibrated area estimator.

Emits ONLY the talc artifact (mask, fraction, overlay). Sulfide/gray channels
are internal auxiliary supervision and are never surfaced (the user maintains a
separate sulfide model).
"""
from __future__ import annotations

from .config import Config, load_config
from .constants import CLASS_TALC, NUM_CLASSES, VALID_CLASSES
from .folds import assign_split, specimen_id, specimen_map
from .scale import parse_magnification, scale_factor
from .seed import set_seed

__all__ = [
    "Config",
    "load_config",
    "set_seed",
    "assign_split",
    "specimen_id",
    "specimen_map",
    "parse_magnification",
    "scale_factor",
    "CLASS_TALC",
    "NUM_CLASSES",
    "VALID_CLASSES",
]
