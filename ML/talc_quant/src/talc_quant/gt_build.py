"""Build the training corpus: talc labels from hand polygons + GMM pseudo-labels
for the surrounding phases, all in the canonical profile at canonical scale.

Talc supervision (zone convention, spec §3.2): the hand-drawn YOLO-seg polygons
are filled solid — small bright inclusions inside a talc zone stay talc. Outside
the polygons, a 1-D GMM on L assigns matrix / gray / sulfide only where confident
(rest -> IGNORE), so uncertain pixels never corrupt the loss. Talc pixels always
win over the GMM. Border-connected resin -> BACKGROUND (excluded from the area
denominator).

Outputs under paths.build_dir:
  images/<stem>.jpg   labels/<stem>.png   fda_bank/<n>.jpg
  manifest.json (list of records)   folds.json ({stem: fold})
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from .config import Config
from .constants import (CLASS_BACKGROUND, CLASS_GRAY, CLASS_MATRIX,
                        CLASS_SULFIDE, CLASS_TALC, IGNORE_INDEX, IMAGE_EXTS)
from .folds import assign_folds
from .preprocess import CanonicalPreprocessor, background_mask
from .scale import scale_factor

logger = logging.getLogger(__name__)


@dataclass
class ItemRecord:
    stem: str
    image: str        # relative path under build_dir
    label: str
    has_talc: bool
    sort: str         # talc | ordinary | fine | negative
    source: str       # yolo_seg | part1 | part2
    fold: int


def gmm_pseudolabels(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """uint8 map {MATRIX, SULFIDE, GRAY, IGNORE} from a 3-component GMM on L."""
    l = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    flat = l.reshape(-1, 1)
    rng = np.random.default_rng(0)
    if flat.shape[0] > cfg.data.gmm_max_pixels:
        idx = rng.choice(flat.shape[0], cfg.data.gmm_max_pixels, replace=False)
        sample = flat[idx]
    else:
        sample = flat
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(sample)
    order = np.argsort(gmm.means_.ravel())  # dark -> mid -> bright
    comp_to_class = {int(order[0]): CLASS_MATRIX,
                     int(order[1]): CLASS_GRAY,
                     int(order[2]): CLASS_SULFIDE}
    proba = gmm.predict_proba(flat)
    conf = proba.max(axis=1)
    comp = proba.argmax(axis=1)
    out = np.full(flat.shape[0], IGNORE_INDEX, dtype=np.uint8)
    keep = conf >= cfg.data.gmm_conf
    mapped = np.array([comp_to_class[int(c)] for c in comp], dtype=np.uint8)
    out[keep] = mapped[keep]
    out = out.reshape(l.shape)
    out[(out == CLASS_SULFIDE) & (l < cfg.data.sulfide_min_l)] = IGNORE_INDEX
    return out


def polygons_to_talc(label_txt: Path, h: int, w: int, min_px: int) -> np.ndarray:
    """Fill YOLO-seg normalized polygons into a solid boolean talc mask."""
    mask = np.zeros((h, w), np.uint8)
    if not label_txt.exists():
        return mask.astype(bool)
    for line in label_txt.read_text().splitlines():
        vals = line.split()
        if len(vals) < 1 + min_px:
            continue
        pts = np.array(vals[1:], np.float64).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def compose_label(bgr: np.ndarray, talc: np.ndarray, cfg: Config) -> np.ndarray:
    """GMM phases + talc override + border-resin background -> uint8 label map."""
    label = gmm_pseudolabels(bgr, cfg)
    label[background_mask(bgr)] = CLASS_BACKGROUND
    label[talc] = CLASS_TALC   # talc always wins
    return label


def _resample(bgr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return bgr
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=interp)


def _iter_folder_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def build_dataset(cfg: Config) -> list[ItemRecord]:
    """Full build. Returns the manifest (also written to disk)."""
    pp = CanonicalPreprocessor(cfg.paths.sulfide_preprocess)
    out = cfg.paths.build_dir
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "fda_bank").mkdir(parents=True, exist_ok=True)

    records: list[ItemRecord] = []

    # --- talc-labeled images (yolo_seg) ---
    talc_stems: list[str] = []
    for split in ("train", "val"):
        img_dir = cfg.paths.yolo_seg / "images" / split
        lbl_dir = cfg.paths.yolo_seg / "labels" / split
        for img_path in _iter_folder_images(img_dir):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            factor = scale_factor(img_path.name, cfg.scale)
            bgr = _resample(pp(bgr), factor)
            h, w = bgr.shape[:2]
            talc = polygons_to_talc(lbl_dir / (img_path.stem + ".txt"),
                                    h, w, cfg.data.min_talc_polygon_px)
            label = compose_label(bgr, talc, cfg)
            stem = img_path.stem
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), bgr)
            cv2.imwrite(str(out / "labels" / f"{stem}.png"), label)
            records.append(ItemRecord(stem, f"images/{stem}.jpg",
                                      f"labels/{stem}.png", True, "talc",
                                      "yolo_seg", -1))
            talc_stems.append(stem)

    # --- negatives (non-talc sorts) for FP-control training signal ---
    neg_sources = {
        "part1": [cfg.paths.part1 / "run_of_mine_ores",
                  cfg.paths.part1 / "refractory_ores"],
        "part2": [cfg.paths.part2 / "run_of_mine",
                  cfg.paths.part2 / "fine_grained"],
    }
    for source, folders in neg_sources.items():
        for folder in folders:
            sort = "fine" if ("refractory" in folder.name or "fine" in folder.name) \
                else "ordinary"
            imgs = _iter_folder_images(folder)[: cfg.data.neg_per_folder]
            for img_path in imgs:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                factor = scale_factor(img_path.name, cfg.scale)
                bgr = _resample(pp(bgr), factor)
                label = compose_label(bgr, np.zeros(bgr.shape[:2], bool), cfg)
                stem = f"neg_{source}_{img_path.stem}"
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), bgr)
                cv2.imwrite(str(out / "labels" / f"{stem}.png"), label)
                records.append(ItemRecord(stem, f"images/{stem}.jpg",
                                          f"labels/{stem}.png", False, sort,
                                          source, -1))

    # --- FDA bank: panorama tiles in canonical profile ---
    _build_fda_bank(cfg, pp, out)

    # --- grouped folds over talc images; negatives round-robined in ---
    folds = assign_folds(talc_stems, k=cfg.train.folds, seed=cfg.train.seed)
    neg_stems = [r.stem for r in records if not r.has_talc]
    for i, stem in enumerate(neg_stems):
        folds[stem] = i % cfg.train.folds
    for r in records:
        r.fold = folds[r.stem]

    (out / "manifest.json").write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2))
    (out / "folds.json").write_text(json.dumps(folds, indent=2))
    logger.info("built %d items (%d talc, %d neg) -> %s",
                len(records), len(talc_stems), len(neg_stems), out)
    return records


def _build_fda_bank(cfg: Config, pp: CanonicalPreprocessor, out: Path,
                    tile: int = 768, per_pano: int = 6) -> None:
    """Cut a few canonical-profile tiles from each panorama as the FDA style bank."""
    panos = _iter_folder_images(cfg.paths.panoramas)
    rng = np.random.default_rng(0)
    n = 0
    for p in panos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for _ in range(per_pano):
            if h <= tile or w <= tile:
                crop = cv2.resize(img, (tile, tile))
            else:
                y = int(rng.integers(0, h - tile))
                x = int(rng.integers(0, w - tile))
                crop = img[y:y + tile, x:x + tile]
            cv2.imwrite(str(out / "fda_bank" / f"{n:04d}.jpg"), pp(crop))
            n += 1
    logger.info("FDA bank: %d panorama tiles", n)
