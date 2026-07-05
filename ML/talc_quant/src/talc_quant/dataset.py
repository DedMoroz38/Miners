"""Patch dataset over the built corpus: rare-class-aware crops, talc copy-paste,
photometric augmentation, decolorize, and FDA — the data-efficiency levers for
42 talc images (spec §5).

Effective sample count is patches, not images: from ~140 images we draw tens of
thousands of `crop`×`crop` windows. Talc crops are up-sampled (rare_center_p),
and a talc bank supplies copy-paste instances so the rare class is seen often.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .constants import (CLASS_GRAY, CLASS_TALC, IGNORE_INDEX, IMAGENET_MEAN,
                        IMAGENET_STD)
from .fda import FDABank

_MEAN = np.array(IMAGENET_MEAN, np.float32)
_STD = np.array(IMAGENET_STD, np.float32)


def load_manifest(build_dir: Path) -> list[dict]:
    return json.loads((build_dir / "manifest.json").read_text())


def items_for_folds(manifest: list[dict], folds: set[int]) -> list[dict]:
    return [r for r in manifest if r["fold"] in folds]


class PatchDataset(Dataset):
    def __init__(self, items: list[dict], cfg: Config, train: bool = True,
                 seed: int = 0) -> None:
        self.items = items
        self.cfg = cfg
        self.crop = cfg.data.crop
        self.train = train
        self.length = cfg.data.patches_per_epoch if train else cfg.data.val_patches
        self.rng = np.random.default_rng(seed)
        self.build = cfg.paths.build_dir
        self.fda = FDABank(self.build / "fda_bank", cfg.data.fda_beta, seed) \
            if train else None
        self._talc_bank = self._build_talc_bank()

    # --- talc copy-paste bank ---
    def _build_talc_bank(self) -> list[tuple[np.ndarray, np.ndarray]]:
        bank: list[tuple[np.ndarray, np.ndarray]] = []
        for it in self.items:
            if not it["has_talc"]:
                continue
            img = cv2.imread(str(self.build / it["image"]))
            lab = cv2.imread(str(self.build / it["label"]), cv2.IMREAD_GRAYSCALE)
            if img is None or lab is None:
                continue
            ys, xs = np.where(lab == CLASS_TALC)
            if len(ys) < 50:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            bank.append((img[y0:y1 + 1, x0:x1 + 1],
                         lab[y0:y1 + 1, x0:x1 + 1] == CLASS_TALC))
        return bank

    def __len__(self) -> int:
        return self.length

    # --- crop centering ---
    def _crop_center(self, label: np.ndarray) -> tuple[int, int]:
        h, w, c = *label.shape, self.crop
        if self.train and self.rng.random() < self.cfg.data.rare_center_p:
            rare = np.isin(label, [CLASS_TALC, CLASS_GRAY])
            ys, xs = np.where(rare)
            if len(ys) > 0:
                k = self.rng.integers(len(ys))
                return (int(np.clip(ys[k], c // 2, max(c // 2, h - c // 2))),
                        int(np.clip(xs[k], c // 2, max(c // 2, w - c // 2))))
        return (int(self.rng.integers(c // 2, max(c // 2 + 1, h - c // 2))),
                int(self.rng.integers(c // 2, max(c // 2 + 1, w - c // 2))))

    def _paste_talc(self, img: np.ndarray, label: np.ndarray) -> None:
        if not self._talc_bank or self.rng.random() >= self.cfg.data.copy_paste_p:
            return
        pi, pm = self._talc_bank[self.rng.integers(len(self._talc_bank))]
        ph, pw = pm.shape
        H, W = label.shape
        if ph >= H or pw >= W:
            return
        y = int(self.rng.integers(0, H - ph))
        x = int(self.rng.integers(0, W - pw))
        img[y:y + ph, x:x + pw][pm] = pi[pm]
        label[y:y + ph, x:x + pw][pm] = CLASS_TALC

    # --- photometric augmentation ---
    def _augment(self, img: np.ndarray) -> np.ndarray:
        j = self.cfg.data.ab_jitter
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1] = np.clip(lab[..., 1] + self.rng.uniform(-j, j), 0, 255)
        lab[..., 2] = np.clip(lab[..., 2] + self.rng.uniform(-j, j), 0, 255)
        img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        img *= self.rng.uniform(0.8, 1.2)                     # brightness
        img = (img - 128) * self.rng.uniform(0.8, 1.2) + 128  # contrast
        if self.rng.random() < 0.3:
            img += self.rng.normal(0, 8, img.shape)           # noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.rng.random() < self.cfg.data.gray_drop_p:     # decolorize
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return img

    def _read(self, it: dict) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.build / it["image"]))
        lab = cv2.imread(str(self.build / it["label"]), cv2.IMREAD_GRAYSCALE)
        c = self.crop
        if img is None or lab is None:
            return np.zeros((c, c, 3), np.uint8), np.full((c, c), IGNORE_INDEX, np.uint8)
        return img, lab

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        c = self.crop
        it = self.items[self.rng.integers(len(self.items))]
        img, label = self._read(it)
        img = cv2.copyMakeBorder(img, 0, max(0, c - img.shape[0]), 0,
                                 max(0, c - img.shape[1]), cv2.BORDER_REFLECT)
        label = cv2.copyMakeBorder(label, 0, max(0, c - label.shape[0]), 0,
                                   max(0, c - label.shape[1]),
                                   cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
        cy, cx = self._crop_center(label)
        img = img[cy - c // 2:cy + c // 2, cx - c // 2:cx + c // 2].copy()
        label = label[cy - c // 2:cy + c // 2, cx - c // 2:cx + c // 2].copy()

        if self.train:
            self._paste_talc(img, label)
            if self.fda is not None and self.rng.random() < self.cfg.data.fda_p:
                img = self.fda.apply(img)
            img = self._augment(img)
            if self.rng.random() < 0.5:
                img, label = img[:, ::-1].copy(), label[:, ::-1].copy()
            if self.rng.random() < 0.5:
                img, label = img[::-1].copy(), label[::-1].copy()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        return {"image": torch.from_numpy(rgb.transpose(2, 0, 1)),
                "label": torch.from_numpy(label.astype(np.int64))}
