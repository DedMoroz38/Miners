"""Build the 5-class training label map for one image.

Combines GMM pseudo-labels (sulfide/gray/matrix) with the hand talc mask,
which takes priority where present.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import CLASS_TALC
from .data_index import Record
from .preprocess import PreprocessConfig, preprocess
from .pseudolabels import pseudo_label
from .yolo_seg import rasterize_talc


def build_label(
    record: Record,
    pre_cfg: PreprocessConfig = PreprocessConfig(),
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (preprocessed BGR image, uint8 label map {0,1,2,3,255}).

    Sulfide/gray/matrix come from GMM pseudo-labels; talc is rasterised from the
    hand-labelled YOLO-seg polygons and overrides pseudo-labels where present.
    """
    raw = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(record.image_path)
    img = preprocess(raw, pre_cfg)
    label = pseudo_label(img, confidence=confidence)

    if record.talc_label is not None:
        h, w = label.shape
        talc = rasterize_talc(record.talc_label, h, w)
        label[talc > 0] = CLASS_TALC
    return img, label
