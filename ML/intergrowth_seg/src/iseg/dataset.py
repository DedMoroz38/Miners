"""Tile dataset with strong augmentation and class-balanced sampling.

Tiles inherit the slide label (weak supervision). Strong photometric + geometric
augmentation is the main lever for generalising across illumination/polishing on a
small weakly-labelled set. Uses albumentations if installed, else a numpy fallback.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .sources import Item

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

try:  # optional but recommended
    import albumentations as A
    _HAS_ALB = True
except Exception:  # noqa: BLE001
    _HAS_ALB = False


def _alb_pipeline(tile: int):
    return A.Compose([
        A.RandomResizedCrop(size=(tile, tile), scale=(0.6, 1.0), ratio=(0.8, 1.25), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.25, 0.25, p=0.7),
        A.HueSaturationValue(10, 15, 10, p=0.4),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=tile // 8, max_width=tile // 8, p=0.3),
    ])


def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class TileDataset(Dataset):
    def __init__(self, items: list[Item], tile: int = 448, tiles_per_image: int = 8,
                 augment: bool = True, seed: int = 0) -> None:
        self.items = items
        self.tile = tile
        self.length = max(1, len(items) * tiles_per_image)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.alb = _alb_pipeline(tile) if (_HAS_ALB and augment) else None

    def __len__(self) -> int:
        return self.length

    def _read_rgb(self, item: Item) -> np.ndarray | None:
        img = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        t = self.tile
        img = cv2.copyMakeBorder(img, 0, max(0, t - img.shape[0]), 0,
                                 max(0, t - img.shape[1]), cv2.BORDER_REFLECT)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _numpy_aug(self, rgb: np.ndarray) -> np.ndarray:
        t = self.tile
        y = int(self.rng.integers(0, rgb.shape[0] - t + 1))
        x = int(self.rng.integers(0, rgb.shape[1] - t + 1))
        crop = rgb[y:y + t, x:x + t].astype(np.float32)
        crop *= self.rng.uniform(0.75, 1.25)
        crop = (crop - 128) * self.rng.uniform(0.75, 1.25) + 128
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.5:
            crop = crop[:, ::-1]
        if self.rng.random() < 0.5:
            crop = crop[::-1]
        if self.rng.random() < 0.5:
            crop = np.rot90(crop, int(self.rng.integers(4)))
        return np.ascontiguousarray(crop)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx % len(self.items)]
        rgb = self._read_rgb(item)
        if rgb is None:
            return {"image": torch.zeros(3, self.tile, self.tile),
                    "label": torch.tensor(item.label, dtype=torch.long)}
        if self.alb is not None:
            crop = self.alb(image=rgb)["image"]
        elif self.augment:
            crop = self._numpy_aug(rgb)
        else:
            t = self.tile
            crop = rgb[:t, :t]
        return {"image": _to_tensor(crop), "label": torch.tensor(item.label, dtype=torch.long)}


def balanced_sampler(items: list[Item], length: int, tiles_per_image: int = 8) -> WeightedRandomSampler:
    """Inverse-frequency sampling so both ore types are seen equally."""
    labels = np.array([it.label for it in items])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    per_item = (1.0 / np.maximum(counts, 1))[labels]
    weights = np.repeat(per_item, tiles_per_image)[:length]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=length, replacement=True)
