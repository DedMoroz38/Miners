"""End-to-end panorama pipeline: preprocess -> sulfide mask -> intergrowth classes."""
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from ..data_module.preprocessing import PreprocessConfig, preprocess_image
from ..data_module.pseudo_labels import PseudoLabelConfig, sulfide_mask_classical
from ..model_module import ModelFactory
from .report import CLASS_FINE, CLASS_NORMAL, build_metrics_table, make_overlay, save_report
from .sliding_window import predict_class_grid, predict_prob_map

logger = logging.getLogger(__name__)


def _fill_nan_by_neighbours(grid: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Propagate valid values into NaN cells by iterative 3x3 neighbour means."""
    p = grid.astype(np.float32).copy()
    for _ in range(max_iter):
        nan_mask = np.isnan(p)
        if not nan_mask.any():
            break
        filled = np.where(nan_mask, 0.0, p)
        counts = cv2.blur((~nan_mask).astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
        sums = cv2.blur(filled, (3, 3), borderType=cv2.BORDER_REPLICATE)
        grow = nan_mask & (counts > 1e-6)
        p[grow] = (sums / np.maximum(counts, 1e-6))[grow]
    return np.nan_to_num(p, nan=0.5)


class PanoramaPipeline:
    """Loads saved weights and turns one panorama into overlay + metrics table."""

    def __init__(self, cfg, weights_dir: Path, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.pre_cfg = PreprocessConfig.from_cfg(cfg.data.preprocess)
        self.pseudo_cfg = PseudoLabelConfig.from_cfg(cfg.data.pseudo)
        decision_path = weights_dir / "decision.json"
        if not decision_path.is_file():
            raise FileNotFoundError(
                f"{decision_path} not found — train the classifier first")
        self.decision: dict = json.loads(decision_path.read_text())
        self.threshold = float(self.decision["threshold"])
        self.segmenter = self._load_segmenter(weights_dir)
        self.classifiers = self._load_classifiers(weights_dir)

    def _load_segmenter(self, weights_dir: Path):
        if not bool(self.cfg.infer.use_unet):
            return None
        path = weights_dir / "segmenter_best.pt"
        if not path.is_file():
            logger.warning("No U-Net weights at %s; falling back to classical seg", path)
            return None
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        model = ModelFactory(str(self.cfg.model.segmenter.name), self.cfg.model.segmenter)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        logger.info("Segmenter loaded (val IoU=%.3f)", float(ckpt.get("best_metric", -1)))
        return model

    def _load_classifiers(self, weights_dir: Path) -> list:
        models = []
        for path in sorted(weights_dir.glob("classifier_fold*.pt")):
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            model = ModelFactory(str(self.cfg.model.classifier.name), self.cfg.model.classifier)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            models.append(model)
        if not models:
            raise FileNotFoundError(f"No classifier_fold*.pt in {weights_dir}")
        logger.info("Loaded %d classifier fold(s)", len(models))
        return models

    # ------------------------------------------------------------------- run
    def run(self, image_path: Path, out_dir: Path) -> "tuple[np.ndarray, object]":
        """Process one panorama; saves report files and returns (class_map, table)."""
        ic = self.cfg.infer
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        logger.info("Panorama %s: %dx%d", image_path.name, bgr.shape[1], bgr.shape[0])
        pre = preprocess_image(bgr, self.pre_cfg)

        sulfide = self._sulfide_mask(pre)
        class_map = self._classify(pre, sulfide)

        table = build_metrics_table(
            class_map, self.cfg.data.microns_per_pixel and float(self.cfg.data.microns_per_pixel))
        overlay = make_overlay(bgr, class_map, float(ic.overlay_alpha))
        save_report(out_dir, image_path.stem, overlay, class_map, table)
        return class_map, table

    def _sulfide_mask(self, pre: np.ndarray) -> np.ndarray:
        ic = self.cfg.infer
        if self.segmenter is not None:
            prob = predict_prob_map(self.segmenter, pre, int(ic.seg_tile),
                                    int(ic.seg_stride), int(ic.batch_size), self.device)
            mask = (prob >= float(ic.seg_threshold)).astype(np.uint8)
        else:
            mask = (sulfide_mask_classical(pre, self.pseudo_cfg) > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < int(ic.min_component_px))
        mask[np.isin(labels, small[small > 0])] = 0
        logger.info("Sulfide fraction: %.2f%%", 100.0 * mask.mean())
        return mask

    def _classify(self, pre: np.ndarray, sulfide: np.ndarray) -> np.ndarray:
        ic = self.cfg.infer
        h, w = sulfide.shape
        grid = predict_class_grid(self.classifiers, pre, sulfide, int(ic.cls_tile),
                                  int(ic.cls_stride), int(ic.batch_size), self.device)
        # fill cells without evidence from valid neighbours, then upsample
        valid = ~np.isnan(grid)
        if not valid.any():
            logger.warning("No sulfide-bearing tiles found — empty class map")
            return np.zeros((h, w), dtype=np.uint8)
        p_fine_grid = _fill_nan_by_neighbours(grid)
        p_fine = cv2.resize(p_fine_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        class_map = np.zeros((h, w), dtype=np.uint8)
        fine = p_fine >= self.threshold
        class_map[(sulfide > 0) & fine] = CLASS_FINE
        class_map[(sulfide > 0) & ~fine] = CLASS_NORMAL
        self._smooth_small_components(class_map, sulfide, p_fine, int(ic.smooth_components_below_px))
        return class_map

    def _smooth_small_components(self, class_map: np.ndarray, sulfide: np.ndarray,
                                 p_fine: np.ndarray, max_px: int) -> None:
        """One grain = one class for small components (mean prob majority vote)."""
        if max_px <= 0:
            return
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sulfide, connectivity=8)
        mean_p = np.zeros(n, dtype=np.float64)
        counts = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        np.add.at(mean_p, labels.ravel(), p_fine.ravel())
        mean_p /= np.maximum(counts, 1.0)
        small = (counts < max_px)
        small[0] = False
        comp_class = np.where(mean_p >= self.threshold, CLASS_FINE, CLASS_NORMAL).astype(np.uint8)
        target = small[labels]
        class_map[target] = comp_class[labels[target]]
        class_map[sulfide == 0] = 0
