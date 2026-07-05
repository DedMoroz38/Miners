"""Full-image / panorama inference: fold ensemble + TTA + cosine-blended tiling.

Produces a calibrated talc probability map and a valid-area mask for any input,
choosing single-pass (fields) vs tiled (panoramas) by size. Per-tile canonical
preprocessing keeps train and inference in one profile (spec §5/§7).
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import Config
from .constants import (CLASS_BACKGROUND, CLASS_TALC, IMAGENET_MEAN,
                        IMAGENET_STD, NUM_CLASSES)
from .models import build_model
from .preprocess import CanonicalPreprocessor, background_mask
from .tiled import ProbAccumulator, cosine_window, crop_padded, iter_tiles

logger = logging.getLogger(__name__)
_MEAN = np.array(IMAGENET_MEAN, np.float32)
_STD = np.array(IMAGENET_STD, np.float32)
_PAD_MULT = 28


@dataclasses.dataclass
class InferResult:
    talc_prob: np.ndarray   # [H, W] float in [0,1]
    valid: np.ndarray       # [H, W] bool
    entropy: np.ndarray     # [H, W] float (per-pixel softmax entropy)


def load_models(ckpt_paths: list[Path], device: torch.device) -> list[torch.nn.Module]:
    """Rebuild each checkpoint's architecture offline and load its weights."""
    from .config import ModelCfg
    models: list[torch.nn.Module] = []
    for p in ckpt_paths:
        # checkpoints hold only tensors + str/int/float metadata -> safe subset
        state = torch.load(str(p), map_location="cpu", weights_only=True)
        track = state.get("cfg_track", "segformer_b2")
        mcfg = ModelCfg(track=track, n_classes=NUM_CLASSES,
                        pretrained=False, tiny=False)
        model = build_model(mcfg)
        model.load_state_dict(state["model"])
        model.eval().to(device)
        models.append(model)
        logger.info("loaded %s (%s, val_mae=%.4f)", p.name, track,
                    state.get("val_mae", float("nan")))
    return models


def _to_tensor(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)


def _pad(bgr: np.ndarray, m: int = _PAD_MULT) -> tuple[np.ndarray, int, int]:
    h, w = bgr.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        bgr = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return bgr, h, w


@torch.no_grad()
def _probs_one(models: list[torch.nn.Module], bgr: np.ndarray,
               device: torch.device, tta_flips: bool) -> np.ndarray:
    """Ensemble+TTA softmax for one (already-preprocessed) tile -> [C, H, W]."""
    padded, h, w = _pad(bgr)
    views = [padded]
    if tta_flips:
        views += [padded[:, ::-1], padded[::-1]]
    acc = None
    for i, v in enumerate(views):
        t = torch.from_numpy(_to_tensor(np.ascontiguousarray(v))).unsqueeze(0).to(device)
        for m in models:
            pr = torch.softmax(m(t), dim=1)[0].cpu().numpy()
            if i == 1:
                pr = pr[:, :, ::-1]
            elif i == 2:
                pr = pr[:, ::-1, :]
            acc = pr if acc is None else acc + pr
    acc /= (len(views) * len(models))
    return acc[:, :h, :w]


@torch.no_grad()
def predict(image_bgr: np.ndarray, models: list[torch.nn.Module], cfg: Config,
            device: torch.device, is_panorama: bool | None = None,
            preprocess: bool = True) -> InferResult:
    """Talc probability + valid mask for a field or panorama.

    preprocess=True  (deploy): apply the canonical profile per tile/field here.
    preprocess=False (calibration/eval on already-built images): use as-is so the
    profile is applied exactly once, matching the training corpus.
    """
    pp = CanonicalPreprocessor(cfg.paths.sulfide_preprocess) if preprocess else None
    h, w = image_bgr.shape[:2]
    tile = cfg.infer.tile
    if is_panorama is None:
        is_panorama = max(h, w) > 1.5 * tile

    if not is_panorama:
        proc = pp(image_bgr) if pp else image_bgr
        probs = _probs_one(models, proc, device, cfg.infer.tta_flips)
    else:
        stride = max(1, int(tile * (1 - cfg.infer.overlap)))
        window = cosine_window(tile)
        acc = ProbAccumulator(h, w, NUM_CLASSES)
        coords = list(iter_tiles(h, w, tile, stride))
        logger.info("panorama %dx%d: %d tiles (%dpx, stride %d)", w, h,
                    len(coords), tile, stride)
        for y0, x0 in coords:
            crop = crop_padded(image_bgr, y0, x0, tile)
            proc = pp(crop) if pp else crop
            p = _probs_one(models, proc, device, cfg.infer.tta_flips)
            acc.add(p, y0, x0, window)
        probs = acc.result()

    talc_prob = probs[CLASS_TALC]
    bg = probs.argmax(0) == CLASS_BACKGROUND
    valid = ~(bg | background_mask(image_bgr))
    eps = 1e-6
    entropy = -(probs * np.log(probs + eps)).sum(0)
    return InferResult(talc_prob=talc_prob, valid=valid, entropy=entropy)
