"""Step 2: preprocess every indexed image, build classical sulfide pseudo masks
and the tile table for the classifier. Produces:

  derived/cache/{id}.jpg        preprocessed image
  derived/cache/{id}_mask.png   pseudo sulfide mask (255 = sulfide)
  derived/index_cached.csv      index + cache paths + image sulfide fraction
  derived/tiles.csv             (image_id, x, y, sulfide_frac) eligible tiles
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.data_module.indexing import load_index
from src.data_module.preprocessing import PreprocessConfig, preprocess_image
from src.data_module.pseudo_labels import PseudoLabelConfig, sulfide_mask_classical
from src.utils import resolve_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _tiles_for_mask(mask: np.ndarray, size: int, stride: int,
                    min_frac: float) -> list[tuple[int, int, float]]:
    h, w = mask.shape
    integral = cv2.integral((mask > 0).astype(np.uint8))
    out = []
    ys = list(range(0, max(1, h - size + 1), stride))
    xs = list(range(0, max(1, w - size + 1), stride))
    for y in ys:
        for x in xs:
            y1, x1 = min(y + size, h), min(x + size, w)
            s = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            frac = s / float((y1 - y) * (x1 - x))
            if frac >= min_frac:
                out.append((x, y, float(frac)))
    return out


def _process_one(args: tuple) -> tuple[int, float, list[tuple[int, int, float]]] | None:
    (image_id, path, cache_dir, pre_cfg, pseudo_cfg, size, stride, min_frac) = args
    img_out = cache_dir / f"{image_id:05d}.jpg"
    mask_out = cache_dir / f"{image_id:05d}_mask.png"
    if img_out.exists() and mask_out.exists():
        mask = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logging.getLogger(__name__).error("Unreadable image: %s", path)
            return None
        pre = preprocess_image(bgr, pre_cfg)
        mask = sulfide_mask_classical(pre, pseudo_cfg)
        cv2.imwrite(str(img_out), pre, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_out), mask)
    return image_id, float((mask > 0).mean()), _tiles_for_mask(mask, size, stride, min_frac)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the preprocessed cache, pseudo masks and tiles table."""
    setup_logging()
    set_seed(int(cfg.seed))
    derived = resolve_path(cfg.paths.derived)
    cache_dir = derived / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = load_index(derived / "index.csv")
    pre_cfg = PreprocessConfig.from_cfg(cfg.data.preprocess)
    pseudo_cfg = PseudoLabelConfig.from_cfg(cfg.data.pseudo)
    t = cfg.data.tiles
    jobs = [(i, row["path"], cache_dir, pre_cfg, pseudo_cfg,
             int(t.size), int(t.stride), float(t.min_sulfide_frac))
            for i, row in df.iterrows()]

    tile_rows: list[dict] = []
    fracs: dict[int, float] = {}
    workers = int(cfg.train_segmenter.num_workers) or 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in tqdm(pool.map(_process_one, jobs, chunksize=4), total=len(jobs),
                        desc="pseudo masks"):
            if res is None:
                continue
            image_id, frac, tiles = res
            fracs[image_id] = frac
            tile_rows += [{"image_id": image_id, "x": x, "y": y, "sulfide_frac": f}
                          for x, y, f in tiles]

    df["image_id"] = df.index
    df["cache_image"] = [f"{i:05d}.jpg" for i in df.index]
    df["cache_mask"] = [f"{i:05d}_mask.png" for i in df.index]
    df["sulfide_frac"] = df.index.map(fracs).fillna(0.0)
    df.to_csv(derived / "index_cached.csv", index=False)
    tiles_df = pd.DataFrame(tile_rows).sort_values(["image_id", "y", "x"])
    tiles_df.to_csv(derived / "tiles.csv", index=False)
    logger.info("Cached %d images; %d eligible tiles; mean sulfide frac %.3f",
                len(fracs), len(tiles_df), df["sulfide_frac"].mean())


if __name__ == "__main__":
    main()
