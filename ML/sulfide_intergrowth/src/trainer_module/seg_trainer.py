"""Trainer for the sulfide U-Net (self-training on classical pseudo masks)."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_module.augmentation import seg_train_transform
from ..data_module.seg_dataset import SegTileDataset
from ..model_module import ModelFactory

logger = logging.getLogger(__name__)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss on sigmoid probabilities."""
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


@torch.no_grad()
def _iou(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return inter / union if union > 0 else 1.0


class SegTrainer:
    """Trains ResNetUNet on pseudo-labelled tiles; saves best-IoU checkpoint."""

    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = ModelFactory(str(cfg.model.segmenter.name), cfg.model.segmenter)
        self.model.to(device)

    def _loaders(self, frame: pd.DataFrame, cache_dir: Path) -> tuple[DataLoader, DataLoader]:
        tc = self.cfg.train_segmenter
        tile = int(self.cfg.data.tiles.size)
        rng = np.random.default_rng(int(self.cfg.seed))
        val_n = max(1, int(len(frame) * float(tc.val_fraction)))
        val_idx = rng.choice(len(frame), size=val_n, replace=False)
        is_val = np.zeros(len(frame), dtype=bool)
        is_val[val_idx] = True
        train_ds = SegTileDataset(frame[~is_val], cache_dir, tile,
                                  int(tc.max_tiles_per_image),
                                  transform=seg_train_transform(tile), train=True)
        val_ds = SegTileDataset(frame[is_val], cache_dir, tile, 2, train=False)
        common = dict(num_workers=int(tc.num_workers), pin_memory=True,
                      persistent_workers=int(tc.num_workers) > 0)
        train_dl = DataLoader(train_ds, batch_size=int(tc.batch_size), shuffle=True, **common)
        val_dl = DataLoader(val_ds, batch_size=int(tc.batch_size), shuffle=False, **common)
        return train_dl, val_dl

    def fit(self, frame: pd.DataFrame, cache_dir: Path, weights_dir: Path) -> Path:
        """Train and persist the best checkpoint.

        Returns:
            Path to the saved best checkpoint.
        """
        tc = self.cfg.train_segmenter
        train_dl, val_dl = self._loaders(frame, cache_dir)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(tc.lr),
                                weight_decay=float(tc.weight_decay))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tc.epochs))
        use_amp = bool(tc.amp) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        bce = nn.BCEWithLogitsLoss()
        best_iou, best_path = -1.0, weights_dir / "segmenter_best.pt"
        weights_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(int(tc.epochs)):
            self.model.train()
            losses = []
            for x, y in tqdm(train_dl, desc=f"seg epoch {epoch}", leave=False):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    logits = self.model(x)
                    loss = bce(logits, y) + dice_loss(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()
            val_iou = self._validate(val_dl)
            logger.info("seg epoch %d: loss=%.4f val_iou=%.4f", epoch,
                        float(np.mean(losses)), val_iou)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_iou": val_iou,
                "best_metric": max(best_iou, val_iou),
                "model_cfg": OmegaConf.to_container(self.cfg.model.segmenter, resolve=True),
            }
            # overwrite the rolling last-epoch checkpoint every epoch
            torch.save(ckpt, weights_dir / "segmenter_last.pt")
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(ckpt, best_path)
        logger.info("Segmenter done: best val IoU=%.4f -> %s", best_iou, best_path)
        return best_path

    @torch.no_grad()
    def _validate(self, val_dl: DataLoader) -> float:
        self.model.eval()
        ious = []
        for x, y in val_dl:
            x, y = x.to(self.device), y.to(self.device)
            ious.append(_iou(self.model(x), y))
        return float(np.mean(ious)) if ious else 0.0
