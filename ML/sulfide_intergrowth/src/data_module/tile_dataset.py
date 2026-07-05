"""Tile dataset for the intergrowth-type classifier.

Every tile inherits the IMAGE-level label of its slide folder (weak
supervision). Only tiles with enough sulfide area are eligible — a background
tile carries no intergrowth information.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentation import to_tensor_chw

logger = logging.getLogger(__name__)


class ClsTileDataset(Dataset):
    """Classification tiles indexed by a tiles table (image_id, x, y, frac)."""

    def __init__(self, tiles: pd.DataFrame, images: pd.DataFrame, cache_dir: Path,
                 tile: int, transform=None) -> None:
        """
        Args:
            tiles: rows (image_id, x, y, sulfide_frac) from make_pseudo_masks.
            images: index frame with cache_image paths, indexed by image_id.
            cache_dir: derived cache directory.
            tile: tile side in px.
            transform: albumentations transform or None.
        """
        self.tiles = tiles.reset_index(drop=True)
        self.images = images
        self.cache_dir = Path(cache_dir)
        self.tile = tile
        self.transform = transform
        self._cache_key: str | None = None
        self._cache_img: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tiles)

    def _get_image(self, cache_image: str) -> np.ndarray:
        # tiles are grouped by image in the table; a 1-slot cache avoids
        # re-decoding the same 4160x2768 JPEG for consecutive tiles
        if self._cache_key != cache_image:
            img = cv2.imread(str(self.cache_dir / cache_image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Missing cached image: {cache_image}")
            self._cache_key, self._cache_img = cache_image, img
        assert self._cache_img is not None
        return self._cache_img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t_row = self.tiles.iloc[idx]
        img_row = self.images.loc[int(t_row["image_id"])]
        t = self.tile
        tile_rel = t_row.get("tile_path")
        if isinstance(tile_rel, str) and tile_rel:
            # fast path: pre-cut crop written by make_pseudo_masks.py
            patch = cv2.imread(str(self.cache_dir / tile_rel), cv2.IMREAD_COLOR)
            if patch is None:
                raise FileNotFoundError(f"Missing tile crop: {tile_rel}")
            patch = patch[:t, :t]
        else:
            image = self._get_image(str(img_row["cache_image"]))
            x0, y0 = int(t_row["x"]), int(t_row["y"])
            patch = image[y0:y0 + t, x0:x0 + t]
        if patch.shape[0] != t or patch.shape[1] != t:
            patch = cv2.copyMakeBorder(patch, 0, t - patch.shape[0], 0, t - patch.shape[1],
                                       cv2.BORDER_REFLECT)
        if self.transform is not None:
            patch = self.transform(image=patch)["image"]
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(to_tensor_chw(rgb))
        y = torch.tensor(int(img_row["label"]), dtype=torch.long)
        return x, y


def make_balanced_sampler(tiles: pd.DataFrame, images: pd.DataFrame,
                          tiles_per_image: int) -> WeightedRandomSampler:
    """Sampler balancing both classes and per-image tile counts.

    Weight of a tile = class_weight / n_tiles_of_its_image, so every image
    contributes comparably per epoch and classes are seen 50/50.
    """
    labels = images.loc[tiles["image_id"].to_numpy(), "label"].to_numpy()
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_w = class_counts.sum() / np.maximum(class_counts, 1.0)
    per_image = tiles.groupby("image_id").size()
    img_n = per_image.loc[tiles["image_id"]].to_numpy().astype(np.float64)
    weights = class_w[labels] / img_n
    num_samples = int(tiles_per_image * len(per_image))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=num_samples, replacement=True)


def top_tiles_per_image(tiles: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Deterministic evaluation tiles: up to `cap` highest-sulfide tiles per image."""
    return (tiles.sort_values(["image_id", "sulfide_frac"], ascending=[True, False])
                 .groupby("image_id").head(cap).reset_index(drop=True))
