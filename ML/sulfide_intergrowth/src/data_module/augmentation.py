"""Albumentations pipelines for both training stages.

Falls back to light numpy-only augmentation if albumentations is missing, so
smoke tests do not require the full dependency stack.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    _HAS_ALB = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_ALB = False
    logger.warning("albumentations not installed; using minimal flip augmentation")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def seg_train_transform(size: int):
    """Geometric + photometric augmentation for segmenter training."""
    if not _HAS_ALB:
        return _FallbackFlip()
    return A.Compose([
        A.RandomCrop(size, size, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def cls_train_transform(size: int):
    """Strong augmentation for the tile classifier."""
    if not _HAS_ALB:
        return _FallbackFlip()
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.9, 1.11)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
        A.HueSaturationValue(8, 12, 8, p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(p=0.2),
    ])


class _FallbackFlip:
    """Minimal flip-only augmentation used when albumentations is absent."""

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1] if mask is not None else None
        if rng.random() < 0.5:
            image = image[::-1]
            mask = mask[::-1] if mask is not None else None
        out = {"image": np.ascontiguousarray(image)}
        if mask is not None:
            out["mask"] = np.ascontiguousarray(mask)
        return out


def to_tensor_chw(image_rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB -> float32 CHW, ImageNet-normalized."""
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))
