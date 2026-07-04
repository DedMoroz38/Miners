"""Shared utilities: seeding, logging, device and path resolution."""
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the project root (ML/sulfide_intergrowth)."""
    return _PROJECT_ROOT


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(requested: str = "auto") -> torch.device:
    """Pick a torch device.

    Args:
        requested: "auto", "cuda", "cpu" or "mps".

    Returns:
        The selected torch device.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, streaming to stdout."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)
