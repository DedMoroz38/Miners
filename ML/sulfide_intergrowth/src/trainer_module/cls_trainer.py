"""K-fold trainer for the intergrowth-type tile classifier.

Honest metric contract: pixel GT does not exist, the only ground truth is the
image-level label. Therefore F1 is computed at IMAGE level: tile probabilities
are soft-voted into an image probability; StratifiedGroupKFold over slide
groups prevents leakage; the decision threshold is tuned on OOF predictions
and the fold ensemble is finally checked on the untouched slide holdout.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_module.augmentation import cls_train_transform
from ..data_module.tile_dataset import ClsTileDataset, make_balanced_sampler, top_tiles_per_image
from ..model_module import ModelFactory

logger = logging.getLogger(__name__)

EVAL_TILES_CAP = 24


class ClassifierTrainer:
    """Trains N folds, tunes the OOF threshold, evaluates the holdout ensemble."""

    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------ folds
    def fit(self, images: pd.DataFrame, tiles: pd.DataFrame, cache_dir: Path,
            weights_dir: Path) -> dict:
        """Run the full K-fold protocol.

        Args:
            images: index frame (image_id-indexed) with label/group/split/cache_image.
            tiles: eligible tiles table.
            cache_dir: derived cache dir.
            weights_dir: output dir for fold weights + decision.json.

        Returns:
            Summary dict with OOF and holdout F1.
        """
        tc = self.cfg.train_classifier
        weights_dir.mkdir(parents=True, exist_ok=True)
        trainval = images[images["split"] == "trainval"]
        holdout = images[images["split"] == "holdout"]
        skf = StratifiedGroupKFold(n_splits=int(tc.n_folds), shuffle=True,
                                   random_state=int(self.cfg.seed))
        ids = trainval.index.to_numpy()
        oof_prob = pd.Series(np.nan, index=trainval.index)
        fold_paths: list[Path] = []

        for fold, (tr, va) in enumerate(skf.split(ids, trainval["label"], trainval["group"])):
            tr_ids, va_ids = ids[tr], ids[va]
            logger.info("fold %d: train %d imgs / val %d imgs", fold, len(tr_ids), len(va_ids))
            model = self._train_one_fold(fold, tr_ids, va_ids, images, tiles, cache_dir)
            path = weights_dir / f"classifier_fold{fold}.pt"
            torch.save({
                "fold": fold,
                "model_state_dict": model.state_dict(),
                "model_cfg": OmegaConf.to_container(self.cfg.model.classifier, resolve=True),
            }, path)
            fold_paths.append(path)
            oof_prob.loc[va_ids] = self._image_probs(model, va_ids, images, tiles, cache_dir)
            del model
            torch.cuda.empty_cache() if self.device.type == "cuda" else None

        thr, oof_f1 = self._tune_threshold(trainval["label"], oof_prob)
        logger.info("OOF image-level F1=%.4f @ thr=%.3f", oof_f1, thr)
        hold_f1 = self._eval_holdout(fold_paths, holdout, images, tiles, cache_dir, thr)
        decision = {
            "threshold": float(thr),
            "oof_f1": float(oof_f1),
            "holdout_f1": float(hold_f1),
            "tile": int(self.cfg.data.tiles.size),
            "min_sulfide_frac": float(self.cfg.data.tiles.min_sulfide_frac),
            "n_folds": int(tc.n_folds),
            "backbone": str(self.cfg.model.classifier.backbone),
        }
        (weights_dir / "decision.json").write_text(json.dumps(decision, indent=2))
        logger.info("HOLDOUT image-level F1=%.4f | decision.json saved", hold_f1)
        return decision

    # ------------------------------------------------------------- fold train
    def _train_one_fold(self, fold: int, tr_ids: np.ndarray, va_ids: np.ndarray,
                        images: pd.DataFrame, tiles: pd.DataFrame,
                        cache_dir: Path) -> nn.Module:
        tc = self.cfg.train_classifier
        tile = int(self.cfg.data.tiles.size)
        model = ModelFactory(str(self.cfg.model.classifier.name), self.cfg.model.classifier)
        model.to(self.device)
        tr_tiles = tiles[tiles["image_id"].isin(tr_ids)].reset_index(drop=True)
        sampler = make_balanced_sampler(tr_tiles, images, int(tc.tiles_per_image))
        train_ds = ClsTileDataset(tr_tiles, images, cache_dir, tile,
                                  transform=cls_train_transform(tile))
        train_dl = DataLoader(train_ds, batch_size=int(tc.batch_size), sampler=sampler,
                              num_workers=int(tc.num_workers), pin_memory=True,
                              persistent_workers=int(tc.num_workers) > 0)
        opt = torch.optim.AdamW(
            model.param_groups(float(tc.lr), float(tc.backbone_lr_mult)),
            weight_decay=float(tc.weight_decay))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tc.epochs))
        use_amp = bool(tc.amp) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        crit = nn.CrossEntropyLoss(label_smoothing=float(tc.label_smoothing))
        best_f1, best_state = -1.0, None

        for epoch in range(int(tc.epochs)):
            model.train()
            losses = []
            for x, y in tqdm(train_dl, desc=f"fold{fold} ep{epoch}", leave=False):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()
            probs = self._image_probs(model, va_ids, images, tiles, cache_dir)
            val_f1 = f1_score(images.loc[va_ids, "label"], (probs >= 0.5).astype(int))
            logger.info("fold %d ep %d: loss=%.4f val_img_f1=%.4f",
                        fold, epoch, float(np.mean(losses)), val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def _image_probs(self, model: nn.Module, image_ids: np.ndarray,
                     images: pd.DataFrame, tiles: pd.DataFrame,
                     cache_dir: Path) -> np.ndarray:
        """Soft-vote deterministic top tiles -> P(fine) per image."""
        tc = self.cfg.train_classifier
        tile = int(self.cfg.data.tiles.size)
        model.eval()
        sub = tiles[tiles["image_id"].isin(image_ids)]
        sub = top_tiles_per_image(sub, EVAL_TILES_CAP)
        ds = ClsTileDataset(sub, images, cache_dir, tile)
        dl = DataLoader(ds, batch_size=int(tc.batch_size), shuffle=False,
                        num_workers=int(tc.num_workers))
        probs: list[np.ndarray] = []
        for x, _ in dl:
            x = x.to(self.device)
            p = torch.softmax(model(x), dim=1)[:, 1]
            if bool(tc.tta):
                p = p + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)[:, 1]
                p = p / 2.0
            probs.append(p.cpu().numpy())
        sub = sub.assign(prob=np.concatenate(probs) if probs else np.zeros(0))
        per_img = sub.groupby("image_id")["prob"].mean()
        # images with zero eligible tiles: no sulfide -> undefined; default 0.5
        return per_img.reindex(image_ids).fillna(0.5).to_numpy()

    @staticmethod
    def _tune_threshold(labels: pd.Series, probs: pd.Series) -> tuple[float, float]:
        mask = probs.notna()
        y, p = labels[mask].to_numpy(), probs[mask].to_numpy()
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 181):
            f1 = f1_score(y, (p >= thr).astype(int))
            if f1 > best_f1:
                best_thr, best_f1 = float(thr), float(f1)
        return best_thr, best_f1

    def _eval_holdout(self, fold_paths: list[Path], holdout: pd.DataFrame,
                      images: pd.DataFrame, tiles: pd.DataFrame,
                      cache_dir: Path, thr: float) -> float:
        if holdout.empty:
            logger.warning("Holdout is empty; skipping")
            return float("nan")
        ids = holdout.index.to_numpy()
        acc = np.zeros(len(ids))
        for path in fold_paths:
            model = ModelFactory(str(self.cfg.model.classifier.name), self.cfg.model.classifier)
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            acc += self._image_probs(model, ids, images, tiles, cache_dir)
            del model
        probs = acc / len(fold_paths)
        return float(f1_score(holdout["label"], (probs >= thr).astype(int)))
