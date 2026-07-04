"""Slide-grouped k-fold training with an OOF F1 estimate and a fold ensemble.

Levers for F1: strong pretrained backbone, class-balanced sampling, weighted CE +
label smoothing, cosine LR, AMP, TTA at inference, and threshold tuning on the OOF
predictions. Group k-fold keeps every slide in exactly one fold (no leakage) and
the OOF F1 is an honest held-out estimate on all training slides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import TileDataset, balanced_sampler
from .infer import predict_image
from .metrics import best_threshold, f1_at
from .model import ModelConfig, build_model
from .sources import Item


@dataclass
class TrainConfig:
    folds: int = 5
    epochs: int = 12
    tile: int = 448
    tiles_per_image: int = 8
    batch: int = 24            # ConvNeXt-Small @448 fits V100-32GB
    lr: float = 2e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    backbone: str = "convnext_small"
    amp: bool = True
    num_workers: int = 8


def _class_weights(items: list[Item], device) -> torch.Tensor:
    c = np.bincount([it.label for it in items], minlength=2).astype(np.float64)
    w = c.sum() / np.maximum(c, 1) / 2
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_one(items: list[Item], device, cfg: TrainConfig) -> torch.nn.Module:
    model = build_model(ModelConfig(backbone=cfg.backbone, pretrained=True)).to(device)
    ds = TileDataset(items, tile=cfg.tile, tiles_per_image=cfg.tiles_per_image, augment=True)
    sampler = balanced_sampler(items, len(ds), cfg.tiles_per_image)
    loader = DataLoader(ds, batch_size=cfg.batch, sampler=sampler,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * max(len(loader), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    w = _class_weights(items, device)
    for epoch in range(cfg.epochs):
        model.train()
        run = 0.0
        for batch in tqdm(loader, desc=f"fold-train e{epoch+1}/{cfg.epochs}"):
            img = batch["image"].to(device, non_blocking=True)
            lab = batch["label"].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(img), lab, weight=w,
                                       label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach())
        print(f"  epoch {epoch+1}: loss={run/max(len(loader),1):.4f}")
    return model


def train_kfold(items: list[Item], device, cfg: TrainConfig = TrainConfig()):
    """Train `folds` models; return (models, oof_result, oof_p_fine, oof_y)."""
    y = np.array([it.label for it in items])
    groups = np.array([it.slide_id for it in items])
    skf = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True, random_state=42)

    models: list[torch.nn.Module] = []
    oof_p = np.full(len(items), np.nan)
    for fold, (tr, va) in enumerate(skf.split(items, y, groups)):
        print(f"\n=== fold {fold+1}/{cfg.folds}: train {len(tr)} / val {len(va)} ===")
        model = _train_one([items[i] for i in tr], device, cfg)
        models.append(model)
        for i in tqdm(va, desc=f"fold {fold+1} oof"):
            _, p_fine, *_ = predict_image([model], items[i].path, device, tile=cfg.tile)
            oof_p[i] = p_fine

    mask = ~np.isnan(oof_p)
    oof = best_threshold(oof_p[mask], y[mask])
    print(f"\nOOF F1={oof.f1*100:.1f}%  P={oof.precision*100:.1f}  R={oof.recall*100:.1f}  "
          f"@thr={oof.threshold:.2f}")
    return models, oof, oof_p[mask], y[mask]


@torch.no_grad()
def evaluate(models, items: list[Item], device, thr: float, tile: int = 448):
    """Ensemble+TTA F1 on a held-out item list at a fixed threshold."""
    p = np.array([predict_image(models, it.path, device, tile=tile)[1] for it in tqdm(items, desc="test")])
    y = np.array([it.label for it in items])
    return f1_at(p, y, thr), p, y


@torch.no_grad()
def _val_probs(model, items, device, tile, tta=False, progress=False):
    """Image-level p_fine over items (single forward, non-overlapping tiles = fast)."""
    it = tqdm(items, desc="val", leave=False) if progress else items
    p = np.array([predict_image([model], s.path, device, tile=tile, stride=tile, tta=tta)[1] for s in it])
    y = np.array([s.label for s in items])
    return p, y


def train_single(train_items: list[Item], val_items: list[Item], device,
                 cfg: TrainConfig = TrainConfig()):
    """Plain train/val (no k-fold). Prints loss + val acc/F1 every epoch.

    Returns (model, history, val_p_fine, val_y). Per-epoch val uses a single
    forward over non-overlapping tiles (no TTA) so monitoring stays cheap.
    """
    model = build_model(ModelConfig(backbone=cfg.backbone, pretrained=True)).to(device)
    ds = TileDataset(train_items, tile=cfg.tile, tiles_per_image=cfg.tiles_per_image, augment=True)
    sampler = balanced_sampler(train_items, len(ds), cfg.tiles_per_image)
    loader = DataLoader(ds, batch_size=cfg.batch, sampler=sampler,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * max(len(loader), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    w = _class_weights(train_items, device)

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        run = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            img = batch["image"].to(device, non_blocking=True)
            lab = batch["label"].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(img), lab, weight=w, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss.detach())
        loss_ep = run / max(len(loader), 1)

        p, y = _val_probs(model, val_items, device, cfg.tile)
        half = f1_at(p, y, 0.5)
        best = best_threshold(p, y)
        acc = (half.tp + half.tn) / max(len(y), 1)
        history.append({"loss": loss_ep, "val_acc": acc, "val_f1": half.f1,
                        "val_f1_best": best.f1, "val_thr": best.threshold})
        print(f"epoch {epoch+1}: loss={loss_ep:.4f} | val_acc={acc*100:.1f}% "
              f"val_F1@0.5={half.f1*100:.1f}% | best_F1={best.f1*100:.1f}% @thr={best.threshold:.2f} "
              f"(P={best.precision*100:.0f}/R={best.recall*100:.0f})")
    p, y = _val_probs(model, val_items, device, cfg.tile)
    return model, history, p, y
