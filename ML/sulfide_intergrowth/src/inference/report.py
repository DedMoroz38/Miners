"""Deliverables: colour overlay and quantitative area-fraction table.

Class map convention: 0 = background, 1 = normal intergrowths (green),
2 = fine intergrowths (red). Talc is out of scope by design (a separate model
will overlay it later, blue is reserved).
"""
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_BG, CLASS_NORMAL, CLASS_FINE = 0, 1, 2
COLOR_NORMAL_BGR = (0, 200, 0)
COLOR_FINE_BGR = (0, 0, 220)


def make_overlay(bgr: np.ndarray, class_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend class colours over the original panorama."""
    overlay = bgr.copy()
    color = np.zeros_like(bgr)
    color[class_map == CLASS_NORMAL] = COLOR_NORMAL_BGR
    color[class_map == CLASS_FINE] = COLOR_FINE_BGR
    hit = class_map > 0
    overlay[hit] = cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0.0)[hit]
    return overlay


def build_metrics_table(class_map: np.ndarray,
                        microns_per_pixel: float | None = None) -> pd.DataFrame:
    """Area fractions of sulfides overall and per intergrowth type.

    Args:
        class_map: uint8 HxW map (0/1/2).
        microns_per_pixel: physical scale; adds absolute mm^2 columns when set.

    Returns:
        Tidy DataFrame, one row per quantity.
    """
    total_px = int(class_map.size)
    normal_px = int((class_map == CLASS_NORMAL).sum())
    fine_px = int((class_map == CLASS_FINE).sum())
    sulfide_px = normal_px + fine_px

    def pct(x: int, base: int) -> float:
        return 100.0 * x / base if base > 0 else 0.0

    rows = [
        {"metric": "Общая доля сульфидов, % площади", "value": pct(sulfide_px, total_px)},
        {"metric": "Обычные срастания, % площади", "value": pct(normal_px, total_px)},
        {"metric": "Тонкие срастания, % площади", "value": pct(fine_px, total_px)},
        {"metric": "Обычные срастания, % от сульфидов", "value": pct(normal_px, sulfide_px)},
        {"metric": "Тонкие срастания, % от сульфидов", "value": pct(fine_px, sulfide_px)},
    ]
    if microns_per_pixel is not None and microns_per_pixel > 0:
        px_mm2 = (microns_per_pixel / 1000.0) ** 2
        rows += [
            {"metric": "Площадь сульфидов, мм²", "value": sulfide_px * px_mm2},
            {"metric": "Обычные срастания, мм²", "value": normal_px * px_mm2},
            {"metric": "Тонкие срастания, мм²", "value": fine_px * px_mm2},
        ]
    df = pd.DataFrame(rows)
    df["value"] = df["value"].round(4)
    return df


CLASS_LABEL = {CLASS_NORMAL: "normal", CLASS_FINE: "fine"}


def build_segments_table(sulfide_mask: np.ndarray, p_fine: np.ndarray,
                         p_sulfide: np.ndarray, threshold: float,
                         microns_per_pixel: float | None = None) -> pd.DataFrame:
    """Per-segment (per sulfide grain) class + confidence table.

    Each connected component of the sulfide mask is one segment. Its class is
    decided by the mean P(fine) over the grain vs `threshold`; `confidence` is
    the probability of the assigned class (0.5..1). `conf_sulfide` is the mean
    segmentation confidence over the grain (1.0 for classical segmentation).

    Args:
        sulfide_mask: uint8 HxW, >0 where sulfide.
        p_fine: float HxW, P(fine intergrowth).
        p_sulfide: float HxW, P(sulfide) from the segmenter.
        threshold: fine-vs-normal decision threshold (from decision.json).
        microns_per_pixel: adds an `area_mm2` column when set.

    Returns:
        DataFrame sorted by descending area, one row per segment.
    """
    mask = (sulfide_mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total_px = int(mask.size)
    if n <= 1:
        cols = ["segment_id", "class", "area_px", "area_pct",
                "confidence", "conf_sulfide", "cx", "cy"]
        return pd.DataFrame(columns=cols)

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    sum_pfine = np.zeros(n, dtype=np.float64)
    sum_psulf = np.zeros(n, dtype=np.float64)
    flat = labels.ravel()
    np.add.at(sum_pfine, flat, p_fine.ravel())
    np.add.at(sum_psulf, flat, p_sulfide.ravel())
    mean_pfine = sum_pfine / np.maximum(areas, 1.0)
    mean_psulf = sum_psulf / np.maximum(areas, 1.0)

    rows = []
    for cid in range(1, n):  # 0 is background
        is_fine = mean_pfine[cid] >= threshold
        cls = CLASS_FINE if is_fine else CLASS_NORMAL
        conf = mean_pfine[cid] if is_fine else 1.0 - mean_pfine[cid]
        row = {
            "segment_id": cid,
            "class": CLASS_LABEL[cls],
            "area_px": int(areas[cid]),
            "area_pct": round(100.0 * areas[cid] / total_px, 5),
            "confidence": round(float(conf), 4),
            "conf_sulfide": round(float(mean_psulf[cid]), 4),
            "cx": int(centroids[cid, 0]),
            "cy": int(centroids[cid, 1]),
        }
        if microns_per_pixel is not None and microns_per_pixel > 0:
            row["area_mm2"] = round(areas[cid] * (microns_per_pixel / 1000.0) ** 2, 6)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("area_px", ascending=False).reset_index(drop=True)


def save_report(out_dir: Path, stem: str, overlay: np.ndarray, class_map: np.ndarray,
                table: pd.DataFrame, segments: pd.DataFrame | None = None) -> None:
    """Persist overlay JPEG, class-map PNG, the metrics table and per-segment CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / f"{stem}_classmap.png"), class_map)
    table.to_csv(out_dir / f"{stem}_metrics.csv", index=False)
    lines = ["| Метрика | Значение |", "|---|---|"]
    lines += [f"| {r.metric} | {r.value} |" for r in table.itertuples()]
    (out_dir / f"{stem}_metrics.md").write_text("\n".join(lines) + "\n")
    if segments is not None:
        segments.to_csv(out_dir / f"{stem}_segments.csv", index=False)
        logger.info("Segments: %d grains -> %s_segments.csv", len(segments), stem)
    logger.info("Report saved to %s (%s_*)", out_dir, stem)
