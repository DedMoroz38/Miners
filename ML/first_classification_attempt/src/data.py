"""Dataset construction: merge part1+part2, stratified split, transforms, loaders."""
import json
import os
import random
from collections import Counter

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import (CLASS_DIRS, CLASS_TO_IDX, CLASSES, DATA_ROOT, IMG_EXTS,
                    SPLITS_PATH)

# Some hackathon JPEGs are slightly truncated; allow PIL to load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --- Index building ----------------------------------------------------------
def build_index():
    """Return list of (relative_path, label_idx) merging both dataset parts."""
    samples = []
    for cls, dirs in CLASS_DIRS.items():
        label = CLASS_TO_IDX[cls]
        for d in dirs:
            abs_d = os.path.join(DATA_ROOT, d)
            if not os.path.isdir(abs_d):
                raise FileNotFoundError(f"Missing data folder: {abs_d}")
            # Non-recursive: this deliberately skips the ignored
            # "Области оталькования" sub-folder inside part1/Оталькованные руды.
            for name in os.listdir(abs_d):
                p = os.path.join(abs_d, name)
                if not os.path.isfile(p):
                    continue
                if os.path.splitext(name)[1].lower() not in IMG_EXTS:
                    continue
                samples.append((os.path.relpath(p, DATA_ROOT), label))
    return samples


def stratified_split(samples, val_frac=0.15, test_frac=0.15, seed=42):
    """Per-class shuffle then split so every class keeps its proportions."""
    by_class = {}
    for rel, label in samples:
        by_class.setdefault(label, []).append(rel)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, items in by_class.items():
        items = sorted(items)          # deterministic base order
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_items = items[:n_test]
        val_items = items[n_test:n_test + n_val]
        train_items = items[n_test + n_val:]
        train += [(p, label) for p in train_items]
        val += [(p, label) for p in val_items]
        test += [(p, label) for p in test_items]
    rng.shuffle(train)
    return train, val, test


def make_splits(val_frac=0.15, test_frac=0.15, seed=42, path=SPLITS_PATH):
    """Build + persist a stratified split so train/eval always agree."""
    samples = build_index()
    train, val, test = stratified_split(samples, val_frac, test_frac, seed)
    payload = {
        "classes": CLASSES,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "train": train,
        "val": val,
        "test": test,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_splits(path=SPLITS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_make_splits(seed=42, val_frac=0.15, test_frac=0.15, path=SPLITS_PATH):
    if os.path.exists(path):
        return load_splits(path)
    return make_splits(val_frac, test_frac, seed, path)


# --- Dataset -----------------------------------------------------------------
def get_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),          # microscopy is flip-invariant
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),  # lighting/contrast robustness
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    resize = int(round(img_size * 1.14))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class OreDataset(Dataset):
    def __init__(self, records, img_size, train, limit=None):
        if limit:
            records = records[:limit]
        self.records = records
        self.tf = get_transforms(img_size, train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rel, label = self.records[i]
        path = os.path.join(DATA_ROOT, rel)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Extremely rare fully-corrupt file: fall back to a black image so a
            # single bad sample never kills a training run.
            img = Image.new("RGB", (256, 256))
        return self.tf(img), label


def class_counts(records):
    c = Counter(label for _, label in records)
    return [c.get(i, 0) for i in range(len(CLASSES))]


def class_weights(records):
    """Inverse-frequency weights for CrossEntropyLoss (imbalanced classes)."""
    counts = class_counts(records)
    total = sum(counts)
    n = len(counts)
    return torch.tensor([total / (n * max(1, c)) for c in counts], dtype=torch.float32)


def make_loader(records, img_size, train, batch_size, num_workers, limit=None):
    ds = OreDataset(records, img_size, train, limit=limit)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,          # MPS does not benefit from pinned memory
        drop_last=False,
        persistent_workers=bool(num_workers),
    )
