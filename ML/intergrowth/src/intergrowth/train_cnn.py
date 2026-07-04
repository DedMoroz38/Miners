"""Train loop and image-level inference for the Method 2 texture CNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import Sample
from .tiles import image_tiles


@dataclass
class CnnTrainConfig:
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.05
    amp: bool = True


def class_weights(samples: list[Sample], n_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights to offset рядовая/труднообогатимая imbalance."""
    counts = np.bincount([s.label for s in samples], minlength=n_classes).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_cnn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
    cfg: CnnTrainConfig = CnnTrainConfig(),
) -> list[float]:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * max(len(loader), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    history: list[float] = []
    for epoch in range(cfg.epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"cnn epoch {epoch+1}/{cfg.epochs}"):
            img = batch["image"].to(device)
            lab = batch["label"].to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = F.cross_entropy(model(img), lab, weight=weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
        history.append(running / max(len(loader), 1))
        print(f"cnn epoch {epoch+1}: loss={history[-1]:.4f}")
    return history


@torch.no_grad()
def predict_image(
    model: torch.nn.Module, image_path, device: torch.device, batch: int = 32
) -> tuple[int, float]:
    """Image label = argmax of tile-probability mean (soft voting). Returns (label, p_fine)."""
    model.eval()
    tiles = image_tiles(image_path)
    if tiles.numel() == 0:
        return 0, 0.0
    probs = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch].to(device)
        probs.append(F.softmax(model(chunk), dim=1).cpu())
    mean_p = torch.cat(probs).mean(0)
    return int(mean_p.argmax()), float(mean_p[1])
