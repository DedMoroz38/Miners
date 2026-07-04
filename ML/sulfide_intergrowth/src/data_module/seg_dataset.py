"""Segmentation dataset: random tiles from cached preprocessed images + pseudo masks."""
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentation import to_tensor_chw

logger = logging.getLogger(__name__)


class SegTileDataset(Dataset):
    """Random (or center) tiles for U-Net training against pseudo masks.

    Expects the cache produced by `make_pseudo_masks.py`: for each index row a
    preprocessed JPEG and a mask PNG under `derived/cache/`.
    """

    def __init__(self, frame: pd.DataFrame, cache_dir: Path, tile: int,
                 tiles_per_image: int, transform=None, train: bool = True) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.tile = tile
        self.tiles_per_image = max(1, tiles_per_image)
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.frame) * self.tiles_per_image

    def _load_pair(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        img_path = self.cache_dir / str(row["cache_image"])
        mask_path = self.cache_dir / str(row["cache_mask"])
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Cache miss for {row['path']}: {img_path}")
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx // self.tiles_per_image]
        image, mask = self._load_pair(row)
        h, w = mask.shape
        t = self.tile
        if self.train:
            rng = np.random.default_rng()
            # bias half of the crops towards sulfide pixels for a useful signal
            if rng.random() < 0.5 and mask.any():
                ys, xs = np.nonzero(mask[::8, ::8])
                k = rng.integers(len(ys))
                cy, cx = int(ys[k] * 8), int(xs[k] * 8)
                y0 = int(np.clip(cy - t // 2, 0, max(0, h - t)))
                x0 = int(np.clip(cx - t // 2, 0, max(0, w - t)))
            else:
                y0 = int(rng.integers(0, max(1, h - t)))
                x0 = int(rng.integers(0, max(1, w - t)))
        else:
            k = idx % self.tiles_per_image
            y0 = (k * 9973) % max(1, h - t)
            x0 = (k * 6151) % max(1, w - t)
        img_t = image[y0:y0 + t, x0:x0 + t]
        msk_t = mask[y0:y0 + t, x0:x0 + t]
        if img_t.shape[0] != t or img_t.shape[1] != t:
            img_t = cv2.copyMakeBorder(img_t, 0, t - img_t.shape[0], 0, t - img_t.shape[1],
                                       cv2.BORDER_REFLECT)
            msk_t = cv2.copyMakeBorder(msk_t, 0, t - msk_t.shape[0], 0, t - msk_t.shape[1],
                                       cv2.BORDER_REFLECT)
        if self.transform is not None and self.train:
            aug = self.transform(image=img_t, mask=msk_t)
            img_t, msk_t = aug["image"], aug["mask"]
        rgb = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(to_tensor_chw(rgb))
        y = torch.from_numpy((msk_t > 127).astype(np.float32))[None]
        return x, y
