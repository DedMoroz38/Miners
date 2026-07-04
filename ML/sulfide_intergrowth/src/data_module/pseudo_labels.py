"""Classical photometric sulfide segmentation -> pseudo ground truth.

Physics: in reflected light, sulfides (pyrrhotite/chalcopyrite/pentlandite,
reflectance ~35-50%) are bright; silicates and talc (<10%) are dark; magnetite
(~20%) is mid-grey. Hysteresis thresholding on illumination-corrected luminance
keeps weak sulfide pixels only when connected to sure-sulfide seeds, which
suppresses magnetite while preserving grain morphology. Holes inside sulfides
are NEVER filled: gangue inclusions are exactly the replacement signal that
separates fine from normal intergrowths.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import luminance


@dataclass(frozen=True)
class PseudoLabelConfig:
    t_strong: int = 160
    t_weak: int = 105
    open_kernel: int = 3
    min_area_px: int = 64

    @staticmethod
    def from_cfg(cfg) -> "PseudoLabelConfig":
        """Build from an OmegaConf node (cfg.data.pseudo)."""
        return PseudoLabelConfig(
            t_strong=int(cfg.t_strong),
            t_weak=int(cfg.t_weak),
            open_kernel=int(cfg.open_kernel),
            min_area_px=int(cfg.min_area_px),
        )


def _hysteresis(lum: np.ndarray, t_strong: int, t_weak: int) -> np.ndarray:
    """Keep weak-threshold components that touch at least one strong pixel."""
    strong = lum >= t_strong
    weak = lum >= t_weak
    n, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(lum, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    strong_labels = np.unique(labels[strong])
    keep[strong_labels[strong_labels > 0]] = True
    return keep[labels]


def sulfide_mask_classical(bgr_preprocessed: np.ndarray, cfg: PseudoLabelConfig) -> np.ndarray:
    """Segment sulfide pixels on a preprocessed BGR image.

    Args:
        bgr_preprocessed: output of `preprocess_image`.
        cfg: threshold configuration.

    Returns:
        uint8 mask, 255 = sulfide, 0 = background. Same HxW as input.
    """
    lum = luminance(bgr_preprocessed)
    mask = _hysteresis(lum, cfg.t_strong, cfg.t_weak)
    mask_u8 = mask.astype(np.uint8)
    if cfg.open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if cfg.min_area_px > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < cfg.min_area_px)
        drop = np.isin(labels, small[small > 0])
        mask_u8[drop] = 0
    return mask_u8 * 255


def sulfide_fraction(mask: np.ndarray) -> float:
    """Area fraction of sulfide pixels in a mask (0..1)."""
    return float((mask > 0).mean())
