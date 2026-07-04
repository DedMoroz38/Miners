"""The two required metrics.

1. Talc-fraction error: |pred talc area fraction - GT talc area fraction|,
   averaged over images that have a hand talc mask (MAE).
2. Intergrowth-type accuracy: predicted dominant intergrowth (normal/fine) vs
   the folder ore-sort label, over non-talc ore images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from .cache import CachedItem
from .classify import RuleThresholds, classify_image
from .constants import CLASS_BACKGROUND, CLASS_TALC
from .grains import GrainParams, extract_grains
from .inference import predict


@dataclass
class MetricResult:
    talc_mae: float
    talc_n: int
    intergrowth_acc: float
    intergrowth_n: int


def _gt_talc_fraction(item: CachedItem) -> float:
    lab = cv2.imread(str(item.label_path), cv2.IMREAD_GRAYSCALE)
    valid = lab != CLASS_BACKGROUND
    return float((lab == CLASS_TALC).sum()) / max(int(valid.sum()), 1)


def evaluate(
    model: torch.nn.Module,
    items: list[CachedItem],
    device: torch.device,
    n_classes: int,
    thresholds: RuleThresholds = RuleThresholds(),
    grain_params: GrainParams = GrainParams(),
) -> MetricResult:
    talc_errs: list[float] = []
    ig_correct = 0
    ig_total = 0

    for item in tqdm(items, desc="evaluate"):
        img = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        class_map, _ = predict(model, img, device, n_classes)

        if item.has_talc:
            valid = class_map != CLASS_BACKGROUND
            pred_frac = float((class_map == CLASS_TALC).sum()) / max(int(valid.sum()), 1)
            talc_errs.append(abs(pred_frac - _gt_talc_fraction(item)))

        if item.sort in ("normal", "fine"):
            grains = extract_grains(class_map, params=grain_params)
            res = classify_image(class_map, grains, thresholds)
            pred = "fine" if res.fine_frac > thresholds.fine_dominance else "normal"
            ig_correct += int(pred == item.sort)
            ig_total += 1

    return MetricResult(
        talc_mae=float(np.mean(talc_errs)) if talc_errs else float("nan"),
        talc_n=len(talc_errs),
        intergrowth_acc=ig_correct / ig_total if ig_total else float("nan"),
        intergrowth_n=ig_total,
    )
