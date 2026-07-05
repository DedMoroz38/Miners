"""Model registry + factory. Every model exposes a unified interface:

    logits = model(pixel_values)   # [B, NUM_CLASSES, H, W] at input resolution

so engine / inference never branch on the backbone. Build from cfg only.
"""
from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from ..config import ModelCfg

MODEL_FACTORY: Dict[str, Callable[[ModelCfg], nn.Module]] = {}


def register_model(name: str):
    def deco(fn: Callable[[ModelCfg], nn.Module]):
        MODEL_FACTORY[name] = fn
        return fn
    return deco


def build_model(cfg: ModelCfg) -> nn.Module:
    """Instantiate the model named cfg.track. Imports the impl modules so their
    @register_model side effects run."""
    from . import dino, segformer  # noqa: F401  (registration side effects)
    if cfg.track not in MODEL_FACTORY:
        raise KeyError(f"unknown model track '{cfg.track}'. "
                       f"registered: {sorted(MODEL_FACTORY)}")
    return MODEL_FACTORY[cfg.track](cfg)


__all__ = ["MODEL_FACTORY", "register_model", "build_model"]
