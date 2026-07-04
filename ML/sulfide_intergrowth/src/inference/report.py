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


def save_report(out_dir: Path, stem: str, overlay: np.ndarray, class_map: np.ndarray,
                table: pd.DataFrame) -> None:
    """Persist overlay JPEG, class-map PNG and the table (CSV + Markdown)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / f"{stem}_classmap.png"), class_map)
    table.to_csv(out_dir / f"{stem}_metrics.csv", index=False)
    lines = ["| Метрика | Значение |", "|---|---|"]
    lines += [f"| {r.metric} | {r.value} |" for r in table.itertuples()]
    (out_dir / f"{stem}_metrics.md").write_text("\n".join(lines) + "\n")
    logger.info("Report saved to %s (%s_*)", out_dir, stem)
