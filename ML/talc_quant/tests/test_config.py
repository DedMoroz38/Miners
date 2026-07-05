"""Config loads, freezes, resolves paths, and honors overrides."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.config import load_config  # noqa: E402


def test_loads_defaults():
    cfg = load_config()
    assert cfg.model.n_classes == 5
    assert cfg.infer.tile == 1024
    assert cfg.report.talc_bgr == (255, 80, 0)  # blue in BGR


def test_frozen():
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.n_classes = 3  # type: ignore[misc]


def test_overrides():
    cfg = load_config(overrides=["model.track=segformer_b3", "train.batch=16"])
    assert cfg.model.track == "segformer_b3"
    assert cfg.train.batch == 16


def test_paths_absolute():
    cfg = load_config()
    assert cfg.paths.yolo_seg.is_absolute()
    assert cfg.paths.sulfide_preprocess.name == "preprocessing.py"
