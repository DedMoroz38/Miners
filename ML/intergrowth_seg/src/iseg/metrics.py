"""F1 for the binary intergrowth task + threshold tuning.

FINE (труднообогатимая) is the positive class. Image-level F1 is the only honest
metric (folder sort is the only ground truth); we tune the decision threshold on
held-out predictions to maximise it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class F1Result:
    f1: float
    precision: float
    recall: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int


def f1_at(p_fine: np.ndarray, y: np.ndarray, thr: float) -> F1Result:
    pred = (p_fine >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return F1Result(f1, prec, rec, thr, tp, fp, fn, tn)


def best_threshold(p_fine: np.ndarray, y: np.ndarray) -> F1Result:
    """Grid-search the threshold that maximises F1 on these predictions."""
    best = f1_at(p_fine, y, 0.5)
    for thr in np.linspace(0.05, 0.95, 91):
        r = f1_at(p_fine, y, float(thr))
        if r.f1 > best.f1:
            best = r
    return best
