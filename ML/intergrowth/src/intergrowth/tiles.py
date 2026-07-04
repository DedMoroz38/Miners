"""Method 2 data: random tiles that inherit the slide-level ore label.

Tile-inherits-label weak supervision (per the confirmed design): each 512x512 tile
takes its parent image's folder label. Noisy but simple; a fine-ore slide is
mostly fine-intergrowth tiles. Mesotexture is a local property, so tiling matches
the geometallurgical texture-classification setup of Pérez-Barnuevo et al. (2018).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import TILE, TILES_PER_IMAGE
from .data import Sample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _normalize(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class TileDataset(Dataset):
    """Random augmented tiles for training the texture CNN."""

    def __init__(
        self,
        samples: list[Sample],
        tile: int = TILE,
        tiles_per_image: int = TILES_PER_IMAGE,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.tile = tile
        self.length = max(1, len(samples) * tiles_per_image)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _read(self, path) -> np.ndarray | None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        t = self.tile
        img = cv2.copyMakeBorder(
            img, 0, max(0, t - img.shape[0]), 0, max(0, t - img.shape[1]), cv2.BORDER_REFLECT
        )
        return img

    def _aug(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img *= self.rng.uniform(0.8, 1.2)            # brightness
        img = (img - 128) * self.rng.uniform(0.8, 1.2) + 128  # contrast
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx % len(self.samples)]
        img = self._read(s.image_path)
        if img is None:
            return {"image": torch.zeros(3, self.tile, self.tile), "label": torch.tensor(s.label)}
        t = self.tile
        y = int(self.rng.integers(0, img.shape[0] - t + 1))
        x = int(self.rng.integers(0, img.shape[1] - t + 1))
        crop = img[y:y + t, x:x + t].copy()
        if self.augment:
            crop = self._aug(crop)
            if self.rng.random() < 0.5:
                crop = crop[:, ::-1].copy()
            if self.rng.random() < 0.5:
                crop = crop[::-1].copy()
        return {"image": _normalize(crop), "label": torch.tensor(s.label, dtype=torch.long)}


def image_tiles(image_path, tile: int = TILE, stride: int | None = None) -> torch.Tensor:
    """Deterministic grid of tiles covering one image, for inference [N,3,t,t]."""
    stride = stride or tile
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return torch.zeros(0, 3, tile, tile)
    img = cv2.copyMakeBorder(
        img, 0, max(0, tile - img.shape[0]), 0, max(0, tile - img.shape[1]), cv2.BORDER_REFLECT
    )
    h, w = img.shape[:2]
    tiles = []
    for y in range(0, max(1, h - tile + 1), stride):
        for x in range(0, max(1, w - tile + 1), stride):
            tiles.append(_normalize(img[y:y + tile, x:x + tile]))
    return torch.stack(tiles) if tiles else torch.zeros(0, 3, tile, tile)
