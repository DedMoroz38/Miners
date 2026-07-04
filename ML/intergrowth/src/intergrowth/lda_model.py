"""Method 1 classifier: discriminant analysis over the minerallurgical indices.

Pérez-Barnuevo et al. (2013) identify the intergrowth type by discriminant
analysis; we use StandardScaler + LinearDiscriminantAnalysis on the aggregated
image features from indices.py. Trains in seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .data import Sample
from .indices import FEATURE_NAMES, image_features


def build_feature_matrix(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the PB2013 feature matrix X and label vector y for samples."""
    feats, labels = [], []
    for s in tqdm(samples, desc="indices"):
        try:
            feats.append(image_features(s.image_path))
            labels.append(s.label)
        except Exception as exc:  # noqa: BLE001 - skip unreadable images
            print(f"skip {s.image_path.name}: {exc}")
    return np.asarray(feats, np.float32), np.asarray(labels, np.int64)


@dataclass
class LdaClassifier:
    scaler: StandardScaler
    lda: LinearDiscriminantAnalysis

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray) -> "LdaClassifier":
        scaler = StandardScaler().fit(x)
        lda = LinearDiscriminantAnalysis().fit(scaler.transform(x), y)
        return LdaClassifier(scaler, lda)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.lda.predict(self.scaler.transform(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.lda.predict_proba(self.scaler.transform(x))

    def coef_table(self) -> list[tuple[str, float]]:
        """Feature importances (LDA coefficients) for interpretability."""
        w = np.ravel(self.lda.coef_)
        return sorted(zip(FEATURE_NAMES, w.tolist()), key=lambda t: -abs(t[1]))
