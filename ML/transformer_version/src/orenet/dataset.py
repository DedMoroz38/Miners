"""Patch dataset with rare-class sampling and talc copy-paste augmentation.

Effective sample count is patches, not images: from ~1200 images we draw tens of
thousands of 512x512 crops. Rare-class sampling and copy-paste target the scarce
talc class (only 42 annotated images).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CachedItem
from .constants import CLASS_GRAY, CLASS_TALC, IGNORE_INDEX

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light photometric jitter (domain robustness). Geometry handled by crops."""
    img = img.astype(np.float32)
    img *= rng.uniform(0.8, 1.2)  # brightness
    img = (img - 128) * rng.uniform(0.8, 1.2) + 128  # contrast
    if rng.random() < 0.3:
        img += rng.normal(0, 8, img.shape)  # noise
    return np.clip(img, 0, 255).astype(np.uint8)


class PatchDataset(Dataset):
    """Random 512x512 crops with rare-class-aware centring and talc copy-paste."""

    def __init__(
        self,
        items: list[CachedItem],
        crop: int = 512,
        length: int = 4000,
        augment: bool = True,
        copy_paste_p: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.items = items
        self.crop = crop
        self.length = length
        self.augment = augment
        self.copy_paste_p = copy_paste_p
        self.rng = np.random.default_rng(seed)
        self._talc_bank = self._build_talc_bank([it for it in items if it.has_talc])

    def _build_talc_bank(self, talc_items: list[CachedItem]) -> list[tuple[np.ndarray, np.ndarray]]:
        bank: list[tuple[np.ndarray, np.ndarray]] = []
        for it in talc_items:
            lab = cv2.imread(str(it.label_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(str(it.image_path), cv2.IMREAD_COLOR)
            if lab is None or img is None:
                continue
            ys, xs = np.where(lab == CLASS_TALC)
            if len(ys) < 50:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            bank.append((img[y0:y1 + 1, x0:x1 + 1], (lab[y0:y1 + 1, x0:x1 + 1] == CLASS_TALC)))
        return bank

    def __len__(self) -> int:
        return self.length

    def _crop_center(self, label: np.ndarray) -> tuple[int, int]:
        h, w = label.shape
        c = self.crop
        # bias toward rare classes (talc, gray) when present
        if self.rng.random() < 0.5:
            rare = np.isin(label, [CLASS_TALC, CLASS_GRAY])
            ys, xs = np.where(rare)
            if len(ys) > 0:
                k = self.rng.integers(len(ys))
                cy = int(np.clip(ys[k], c // 2, h - c // 2))
                cx = int(np.clip(xs[k], c // 2, w - c // 2))
                return cy, cx
        cy = int(self.rng.integers(c // 2, max(c // 2 + 1, h - c // 2)))
        cx = int(self.rng.integers(c // 2, max(c // 2 + 1, w - c // 2)))
        return cy, cx

    def _paste_talc(self, img: np.ndarray, label: np.ndarray) -> None:
        if not self._talc_bank or self.rng.random() >= self.copy_paste_p:
            return
        patch_img, patch_mask = self._talc_bank[self.rng.integers(len(self._talc_bank))]
        ph, pw = patch_mask.shape
        H, W = label.shape
        if ph >= H or pw >= W:
            return
        y = int(self.rng.integers(0, H - ph))
        x = int(self.rng.integers(0, W - pw))
        region_i = img[y:y + ph, x:x + pw]
        region_l = label[y:y + ph, x:x + pw]
        region_i[patch_mask] = patch_img[patch_mask]
        region_l[patch_mask] = CLASS_TALC

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        c = self.crop
        it = self.items[self.rng.integers(len(self.items))]
        img = cv2.imread(str(it.image_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(it.label_path), cv2.IMREAD_GRAYSCALE)
        if img is None or label is None:
            img = np.zeros((c, c, 3), np.uint8)
            label = np.full((c, c), IGNORE_INDEX, np.uint8)

        img = cv2.copyMakeBorder(img, 0, max(0, c - img.shape[0]), 0, max(0, c - img.shape[1]), cv2.BORDER_REFLECT)
        label = cv2.copyMakeBorder(label, 0, max(0, c - label.shape[0]), 0, max(0, c - label.shape[1]), cv2.BORDER_CONSTANT, value=IGNORE_INDEX)

        cy, cx = self._crop_center(label)
        img = img[cy - c // 2:cy + c // 2, cx - c // 2:cx + c // 2].copy()
        label = label[cy - c // 2:cy + c // 2, cx - c // 2:cx + c // 2].copy()

        if self.augment:
            self._paste_talc(img, label)
            img = _augment(img, self.rng)
            if self.rng.random() < 0.5:
                img, label = img[:, ::-1].copy(), label[:, ::-1].copy()
            if self.rng.random() < 0.5:
                img, label = img[::-1].copy(), label[::-1].copy()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(rgb.transpose(2, 0, 1)),
            "label": torch.from_numpy(label.astype(np.int64)),
        }
