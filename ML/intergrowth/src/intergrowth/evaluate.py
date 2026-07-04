"""Evaluate both methods on a held-out test split: accuracy + confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from .constants import CLASS_NAMES, NUM_CLASSES
from .data import Sample
from .lda_model import LdaClassifier, build_feature_matrix
from .train_cnn import predict_image


@dataclass
class EvalResult:
    method: str
    accuracy: float
    balanced_accuracy: float
    confusion: np.ndarray  # [true, pred]
    n: int


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    m = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def _scores(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    conf = _confusion(y_true, y_pred)
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    per_cls = [
        (conf[c, c] / conf[c].sum()) for c in range(NUM_CLASSES) if conf[c].sum() > 0
    ]
    bal = float(np.mean(per_cls)) if per_cls else float("nan")
    return EvalResult(name, acc, bal, conf, len(y_true))


def eval_method1(clf: LdaClassifier, test: list[Sample]) -> EvalResult:
    x, y = build_feature_matrix(test)
    return _scores("Method 1 (PB2013 indices+LDA)", y, clf.predict(x))


def eval_method2(model: torch.nn.Module, test: list[Sample], device: torch.device) -> EvalResult:
    y_true, y_pred = [], []
    for s in tqdm(test, desc="cnn eval"):
        pred, _ = predict_image(model, s.image_path, device)
        y_true.append(s.label)
        y_pred.append(pred)
    return _scores("Method 2 (CNN texture)", np.array(y_true), np.array(y_pred))


def format_result(r: EvalResult) -> str:
    hdr = f"{r.method}: acc={r.accuracy*100:.1f}%  balanced={r.balanced_accuracy*100:.1f}%  (n={r.n})"
    names = [CLASS_NAMES[c] for c in range(NUM_CLASSES)]
    rows = ["            pred_" + "  pred_".join(names)]
    for c in range(NUM_CLASSES):
        rows.append(f"true_{names[c]:<6} " + "  ".join(f"{v:6d}" for v in r.confusion[c]))
    return hdr + "\n" + "\n".join(rows)
