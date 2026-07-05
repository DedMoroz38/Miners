"""Talc area fraction from probabilities + post-hoc calibration to hit ±3%.

fraction = Σ p(talc) over valid pixels / N_valid  (probabilistic classify-and-
count — a soft count is smoother and better-calibrated than argmax pixel counts).

Calibration is fit ONLY on out-of-fold predictions with ≤3 parameters so it
cannot overfit the 42 points:
  * temperature  — scales the talc logit/prob sharpness (variance control);
  * Theil–Sen line pred→GT (robust slope+intercept), OR Forman's adjusted-count
    correction, OR identity. Choice is cfg.calibrate.method.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calib:
    temperature: float = 1.0
    slope: float = 1.0
    intercept: float = 0.0
    method: str = "identity"

    def apply(self, raw_fraction: float) -> float:
        return float(np.clip(self.slope * raw_fraction + self.intercept, 0.0, 1.0))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> "Calib":
        return Calib(**json.loads(Path(path).read_text()))


def soft_fraction(talc_prob: np.ndarray, valid: np.ndarray,
                  temperature: float = 1.0) -> float:
    """Σ (sharpened) talc prob over valid / N_valid.

    Temperature sharpens via a logit-space scale on a 2-class (talc vs rest)
    reduction, which is what actually moves the soft count.
    """
    p = np.clip(talc_prob[valid].astype(np.float64), 1e-6, 1 - 1e-6)
    if abs(temperature - 1.0) > 1e-6:
        logit = np.log(p / (1 - p)) / temperature
        p = 1.0 / (1.0 + np.exp(-logit))
    n = max(int(valid.sum()), 1)
    return float(p.sum() / n)


def theil_sen(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Robust slope+intercept of gt ≈ slope*pred + intercept (median of pairwise
    slopes; median-based intercept). Robust to a few mislabeled outliers."""
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    n = len(pred)
    slopes = [(gt[j] - gt[i]) / (pred[j] - pred[i])
              for i in range(n) for j in range(i + 1, n)
              if abs(pred[j] - pred[i]) > 1e-9]
    if not slopes:
        return 1.0, 0.0
    slope = float(np.median(slopes))
    intercept = float(np.median(gt - slope * pred))
    return slope, intercept


def fit_calibration(pred: np.ndarray, gt: np.ndarray, method: str = "theilsen",
                    temperature: float = 1.0) -> Calib:
    """Fit a Calib from paired OOF (pred, gt) fractions."""
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    if method == "identity" or len(pred) < 3:
        return Calib(temperature, 1.0, 0.0, "identity")
    if method == "theilsen":
        s, b = theil_sen(pred, gt)
        return Calib(temperature, s, b, "theilsen")
    if method == "adjusted_count":
        # Forman: fraction_adj = (CC - fpr) / (tpr - fpr), estimated globally.
        # Approximate tpr/fpr from the OOF fit: slope≈(tpr-fpr), intercept≈fpr.
        s, b = theil_sen(pred, gt)
        s = s if abs(s) > 1e-3 else 1.0
        return Calib(temperature, 1.0 / s, -b / s, "adjusted_count")
    raise ValueError(f"unknown calibration method '{method}'")


def fraction_stats(errs: np.ndarray) -> dict[str, float]:
    """MAE / median / P90 / max / signed-bias summary of a fraction-error array."""
    e = np.asarray(errs, float)
    a = np.abs(e)
    return {
        "mae": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
        "bias": float(e.mean()),
        "n": int(len(e)),
    }
