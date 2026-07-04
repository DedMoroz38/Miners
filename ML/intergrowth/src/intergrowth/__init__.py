"""Intergrowth-type classification (обычные vs тонкие срастания) for non-talc ore.

Two faithful reimplementations, compared on a held-out test split:
  Method 1 — minerallurgical indices + LDA   (Pérez-Barnuevo et al. 2013)
  Method 2 — ImageNet-pretrained CNN texture (Pérez-Barnuevo et al. 2018 paradigm;
             MISIS/LumenStone, Korshunov et al. 2025)

Neither released code; these are reimplementations from the papers, trained on the
user's own folder-labelled data (рядовая = NORMAL, труднообогатимая = FINE).
"""

from __future__ import annotations

from .constants import CLASS_NAMES, FINE, NORMAL, NUM_CLASSES
from .data import Sample, build_samples, group_split

__all__ = [
    "NORMAL",
    "FINE",
    "NUM_CLASSES",
    "CLASS_NAMES",
    "Sample",
    "build_samples",
    "group_split",
]
