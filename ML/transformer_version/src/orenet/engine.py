"""Training / validation loop for the phase segmenter (CUDA + AMP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .constants import IGNORE_INDEX, NUM_CLASSES
from .losses import DiceFocalTversky
from .model import forward_logits


@dataclass
class TrainConfig:
    epochs: int = 20
    steps_per_epoch: int = 250
    lr: float = 6e-5
    weight_decay: float = 0.01
    talc_weight: float = 8.0
    amp: bool = True


def _class_weights(cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    w = torch.ones(NUM_CLASSES, device=device)
    w[3] = cfg.talc_weight  # CLASS_TALC
    return w


@torch.no_grad()
def per_class_iou(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    for batch in loader:
        img = batch["image"].to(device)
        tgt = batch["label"].numpy()
        pred = forward_logits(model, img).argmax(1).cpu().numpy()
        for c in range(NUM_CLASSES):
            valid = tgt != IGNORE_INDEX
            p, t = (pred == c) & valid, (tgt == c) & valid
            inter[c] += np.logical_and(p, t).sum()
            union[c] += np.logical_or(p, t).sum()
    return inter / np.maximum(union, 1)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig = TrainConfig(),
) -> dict[str, list[float]]:
    model.to(device)
    loss_fn = DiceFocalTversky(class_weights=_class_weights(cfg, device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history: dict[str, list[float]] = {"loss": [], "talc_iou": [], "miou": []}
    for epoch in range(cfg.epochs):
        model.train()
        it = iter(train_loader)
        running = 0.0
        for _ in tqdm(range(cfg.steps_per_epoch), desc=f"epoch {epoch+1}/{cfg.epochs}"):
            batch = next(it)
            img = batch["image"].to(device)
            tgt = batch["label"].to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                logits = forward_logits(model, img)
                loss = loss_fn(logits, tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
        iou = per_class_iou(model, val_loader, device)
        history["loss"].append(running / cfg.steps_per_epoch)
        history["talc_iou"].append(float(iou[3]))
        history["miou"].append(float(iou.mean()))
        print(
            f"epoch {epoch+1}: loss={history['loss'][-1]:.4f} "
            f"talc_iou={iou[3]:.3f} miou={iou.mean():.3f}"
        )
    return history
