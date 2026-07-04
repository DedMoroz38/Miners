"""Photometric preprocessing for reflected-light OM images.

Goals: flatten uneven illumination / stitching vignette on panoramas while
PRESERVING absolute reflectance levels (sulfide vs gangue separation relies on
them), plus optional denoising. CLAHE is off by default for the same reason.
"""
from dataclasses import dataclass

import cv2
import numpy as np


DARK_ANCHOR_TARGET = 25.0
BRIGHT_ANCHOR_TARGET = 210.0
BRIGHT_PERCENTILE = 99.8
MIN_ANCHOR_SPREAD = 40.0


@dataclass(frozen=True)
class PreprocessConfig:
    illumination: str = "anchor"        # "none" | "anchor" | "divide"
    illumination_downscale: int = 8
    illumination_sigma: float = 65.0
    denoise: str = "median"             # "none" | "median" | "nlm" (nlm is slow)
    denoise_h: int = 5                  # strength for nlm / kernel hint for median
    clahe_clip: float = 0.0             # 0 disables

    @staticmethod
    def from_cfg(cfg) -> "PreprocessConfig":
        """Build from an OmegaConf node (cfg.data.preprocess)."""
        return PreprocessConfig(
            illumination=str(cfg.illumination),
            illumination_downscale=int(cfg.illumination_downscale),
            illumination_sigma=float(cfg.illumination_sigma),
            denoise=str(cfg.denoise),
            denoise_h=int(cfg.denoise_h),
            clahe_clip=float(cfg.clahe_clip),
        )


def _estimate_flat_field(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Estimate a smooth multiplicative illumination field on a downscaled copy.

    A median pre-filter makes the estimate robust to bright sulfide grains, so
    large sulfide masses are not flattened away.
    """
    ds = max(1, cfg.illumination_downscale)
    small = cv2.resize(gray, (max(8, gray.shape[1] // ds), max(8, gray.shape[0] // ds)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 31)
    small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), cfg.illumination_sigma)
    field = cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(field, 1.0, None)


def normalize_illumination(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Divide L channel by its smooth field, rescaled to keep the global mean."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    field = _estimate_flat_field(lab[:, :, 0], cfg)
    corrected = l_ch * (float(field.mean()) / field)
    lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_exposure_anchor(bgr: np.ndarray) -> np.ndarray:
    """Two-point photometric standardization across differently exposed shots.

    Maps the dark gangue histogram mode to DARK_ANCHOR_TARGET and the bright
    sulfide tail (p99.8) to BRIGHT_ANCHOR_TARGET with one linear transform of
    the L channel. Unlike flat-field division this cannot erase large sulfide
    masses. If the image has no bright tail (no sulfide), only the dark anchor
    is applied as a pure gain.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    hist = cv2.calcHist([lab], [0], None, [256], [0, 256]).ravel()
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 0).ravel()
    dark = float(np.argmax(hist[:120]))
    bright = float(np.percentile(l_ch, BRIGHT_PERCENTILE))
    if bright - dark >= MIN_ANCHOR_SPREAD:
        scale = (BRIGHT_ANCHOR_TARGET - DARK_ANCHOR_TARGET) / (bright - dark)
        corrected = (l_ch - dark) * scale + DARK_ANCHOR_TARGET
    else:
        corrected = l_ch * (DARK_ANCHOR_TARGET / max(dark, 1.0))
    lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Full preprocessing chain: illumination -> denoise -> optional CLAHE.

    Args:
        bgr: input image (uint8, BGR).
        cfg: preprocessing configuration.

    Returns:
        Preprocessed uint8 BGR image of the same shape.
    """
    out = bgr
    if cfg.illumination == "anchor":
        out = normalize_exposure_anchor(out)
    elif cfg.illumination == "divide":
        out = normalize_illumination(out, cfg)
    if cfg.denoise == "median":
        out = cv2.medianBlur(out, 3)
    elif cfg.denoise == "nlm" and cfg.denoise_h > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None, cfg.denoise_h, cfg.denoise_h, 7, 21)
    if cfg.clahe_clip > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(16, 16))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def luminance(bgr: np.ndarray) -> np.ndarray:
    """Return the L channel (uint8) of the LAB representation."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
