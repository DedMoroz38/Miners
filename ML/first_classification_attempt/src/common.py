"""Shared utilities: device selection, seeding, paths."""
import os
import random

import numpy as np
import torch

# --- Paths -------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)                 # first_classification_attempt/
DATA_ROOT = os.path.join(PROJECT_DIR, "data")          # .../data
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")           # checkpoints / reports
SPLITS_PATH = os.path.join(PROJECT_DIR, "splits.json")

# --- Class definition --------------------------------------------------------
# Three merged classes. The talc "Области оталькования" sub-folder is IGNORED
# (it holds the annotated copies of the оталькованные images, not new samples).
# "тонкие" (part2) == "Труднообогатимые руды" (part1) -> hard-to-process ore.
CLASSES = ["ordinary", "hard", "talc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Human-readable labels (for reports).
CLASS_RU = {
    "ordinary": "Рядовые (ordinary)",
    "hard": "Труднообогатимые/тонкие (hard)",
    "talc": "Оталькованные (talc)",
}

# Folders that make up each class (relative to DATA_ROOT), non-recursive.
CLASS_DIRS = {
    "talc": ["part1/Оталькованные руды", "part2/оталькованные"],
    "ordinary": ["part1/Рядовые руды", "part2/рядовые"],
    "hard": ["part1/Труднообогатимые руды", "part2/тонкие"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        # Let unsupported ops fall back to CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
