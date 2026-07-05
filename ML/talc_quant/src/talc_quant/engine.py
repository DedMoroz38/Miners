"""Training loop for one fold: AMP, AdamW, cosine LR with warmup, early stop on
validation talc-fraction MAE (the acceptance metric), best-checkpoint saving.

Validation here is a fast, resize-based estimate of the talc fraction — enough to
rank checkpoints. The honest, tiled, calibrated number is produced later by
calibrate.py. Trains only parameters with requires_grad (track B freezes DINO).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .constants import (CLASS_TALC, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD,
                        VALID_CLASSES)
from .dataset import PatchDataset, items_for_split, load_manifest
from .losses import SegLoss
from .models import build_model

logger = logging.getLogger(__name__)
_MEAN = np.array(IMAGENET_MEAN, np.float32)
_STD = np.array(IMAGENET_STD, np.float32)
_PAD_MULT = 28  # divisible by SegFormer's 4 and DINOv2's 14


def resolve_device(arg: str) -> torch.device:
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def _pad_to_mult(x: np.ndarray, m: int = _PAD_MULT) -> tuple[np.ndarray, int, int]:
    h, w = x.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    return x, h, w


@torch.no_grad()
def _predict_talc_prob(model: torch.nn.Module, bgr: np.ndarray,
                       device: torch.device, eval_long: int = 768) -> np.ndarray:
    """Resize-based full-image talc probability map (H,W) in [0,1]."""
    h0, w0 = bgr.shape[:2]
    scale = eval_long / max(h0, w0)
    small = cv2.resize(bgr, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                       interpolation=cv2.INTER_AREA)
    padded, ph, pw = _pad_to_mult(small)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    logits = model(t)
    prob = torch.softmax(logits, dim=1)[0, CLASS_TALC].cpu().numpy()
    prob = prob[:ph, :pw]
    return cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def val_talc_mae(model: torch.nn.Module, val_items: list[dict], cfg: Config,
                 device: torch.device) -> float:
    """Mean |pred - GT| talc fraction over val images that carry a talc mask."""
    model.eval()
    errs: list[float] = []
    build = cfg.paths.build_dir
    for it in val_items:
        if not it["has_talc"]:
            continue
        bgr = cv2.imread(str(build / it["image"]))
        lab = cv2.imread(str(build / it["label"]), cv2.IMREAD_GRAYSCALE)
        if bgr is None or lab is None:
            continue
        valid = np.isin(lab, VALID_CLASSES)
        nvalid = max(int(valid.sum()), 1)
        gt = float((lab == CLASS_TALC).sum()) / nvalid
        prob = _predict_talc_prob(model, bgr, device)
        pred = float(prob[valid].sum()) / nvalid
        errs.append(abs(pred - gt))
    return float(np.mean(errs)) if errs else float("nan")


def train_model(cfg: Config, out_dir: Path,
                max_steps: int | None = None) -> Path:
    """Train on the TRAIN split, early-stop on the TEST split's talc-MAE.
    Returns the best checkpoint path (out_dir/best.pt)."""
    device = resolve_device(cfg.train.device)
    logger.info("device %s | track %s", device, cfg.model.track)

    manifest = load_manifest(cfg.paths.build_dir)
    train_items = items_for_split(manifest, "train")
    val_items = items_for_split(manifest, "test")
    logger.info("train items %d | test items %d (%d with talc)",
                len(train_items), len(val_items),
                sum(r["has_talc"] for r in val_items))

    ds = PatchDataset(train_items, cfg, train=True, seed=cfg.train.seed)
    dl = DataLoader(ds, batch_size=cfg.train.batch, num_workers=cfg.train.workers,
                    pin_memory=(device.type == "cuda"), drop_last=True)

    model = build_model(cfg.model).to(device)
    loss_fn = SegLoss(cfg.loss).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    steps_per_epoch = max(1, len(ds) // cfg.train.batch)
    total = max_steps or steps_per_epoch * cfg.train.epochs
    warmup = int(cfg.train.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _cosine_warmup(s, total, warmup))
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pt"
    best_mae, since_best, step = float("inf"), 0, 0
    model.train()
    dl_iter = iter(dl)
    while step < total:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)
        img = batch["image"].to(device)
        lab = batch["label"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img)
            loss = loss_fn(logits, lab)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % steps_per_epoch == 0:
            mae = val_talc_mae(model, val_items, cfg, device)
            logger.info("step %d/%d | loss %.4f | test talc-MAE %.4f",
                        step, total, float(loss.detach()), mae)
            model.train()
            if mae < best_mae:
                best_mae, since_best = mae, 0
                torch.save({"model": model.state_dict(), "cfg_track": cfg.model.track,
                            "test_mae": mae}, best_ckpt)
            else:
                since_best += 1
                if since_best >= cfg.train.patience:
                    logger.info("early stop at step %d (best MAE %.4f)", step, best_mae)
                    break
    if not best_ckpt.exists():  # e.g. no test talc images -> save last
        torch.save({"model": model.state_dict(), "cfg_track": cfg.model.track,
                    "test_mae": best_mae}, best_ckpt)
    logger.info("done. best test talc-MAE %.4f -> %s", best_mae, best_ckpt)
    return best_ckpt
