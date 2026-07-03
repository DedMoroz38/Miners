"""Dataset + augmentation for the augmentation experiment.

Everything except the augmentation strategy is identical to the baseline. The
new pieces are:
  * get_transforms(..., policy="strong"): TrivialAugmentWide + geometric flips +
    RandomErasing on top of RandomResizedCrop.
  * make_sampler(): class-balancing WeightedRandomSampler (talc oversampled).
  * build_mixups(): MixUp/CutMix batch collator (applied in the train loop).
"""
import json
import os
import random
from collections import Counter

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2

from common import (CLASS_DIRS, CLASS_TO_IDX, CLASSES, DATA_ROOT, IMG_EXTS,
                    SPLITS_PATH)

# Some hackathon JPEGs are slightly truncated; allow PIL to load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AUG_POLICIES = ["basic", "strong"]


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


# --- Transforms --------------------------------------------------------------
def get_transforms(img_size: int, train: bool, policy: str = "strong",
                   erase_prob: float = 0.25):
    """Train-time augmentation pipeline.

    policy="basic"  -> the baseline pipeline (RRC + flips + rotation + jitter).
    policy="strong" -> RRC + flips + TrivialAugmentWide + RandomErasing.

    Eval is never augmented (deterministic resize + center crop), regardless of
    policy — test-time augmentation is handled separately in evaluate.py.
    """
    if not train:
        resize = int(round(img_size * 1.14))
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if policy == "basic":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if policy == "strong":
        # Flips are always-on and label-safe for microscopy (no canonical
        # orientation). TrivialAugmentWide adds ONE randomly-chosen op per image
        # at a random magnitude (rotate/shear/translate + color/contrast/sharpness)
        # — a parameter-free policy, so nothing to tune on this tiny dataset.
        # RandomErasing (post-normalize) forces reliance on local mineral texture
        # rather than a single dominant region.
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=erase_prob, value="random"),
        ])

    raise ValueError(f"Unknown aug policy '{policy}'. Options: {AUG_POLICIES}")


# --- Dataset -----------------------------------------------------------------
class OreDataset(Dataset):
    def __init__(self, records, img_size, train, policy="strong", erase_prob=0.25):
        self.records = records
        self.tf = get_transforms(img_size, train, policy, erase_prob)

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


# --- Balancing sampler -------------------------------------------------------
def make_sampler(records, seed=42):
    """WeightedRandomSampler giving every CLASS equal expected mass per epoch.

    Per-sample weight = 1 / count(its class), so talc (few samples) is drawn far
    more often than ordinary. Combined with strong augmentation each draw is a
    different view, so this multiplies talc's *effective* diversity instead of
    just duplicating 91 images. Epoch length is kept equal to len(records) for
    comparability with the baseline.
    """
    counts = class_counts(records)
    per_sample = [1.0 / max(1, counts[label]) for _, label in records]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(per_sample, num_samples=len(records),
                                 replacement=True, generator=g)


def make_loader(records, img_size, train, batch_size, num_workers,
                limit=None, policy="strong", use_sampler=False, seed=42,
                erase_prob=0.25, drop_last=False):
    if limit:
        records = records[:limit]
    ds = OreDataset(records, img_size, train, policy=policy, erase_prob=erase_prob)
    sampler = make_sampler(records, seed) if (train and use_sampler) else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,          # MPS does not benefit from pinned memory
        drop_last=drop_last,
        persistent_workers=bool(num_workers),
    )


# --- MixUp / CutMix ----------------------------------------------------------
def build_mixups(num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
    """Return a (mixup, cutmix) pair of v2 batch transforms, or (None, None).

    Applied per-batch in the training loop (see train.py). MixUp blends two
    images/labels linearly; CutMix pastes a rectangular patch. Both turn the
    integer label into a soft (probability) target, which regularizes decision
    boundaries — helpful for the ordinary<->hard confusion. alpha values follow
    the EfficientNetV2 recipe (mixup 0.2 is U-shaped: usually one image dominates,
    so labels stay mostly clean).
    """
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None, None
    mixup = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes) if mixup_alpha > 0 else None
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes) if cutmix_alpha > 0 else None
    return mixup, cutmix
