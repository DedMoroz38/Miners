"""Deep Zoom tiling: turn a (possibly gigapixel) source image into a DZI pyramid
so the frontend streams only the visible tiles via OpenSeadragon.

libvips (pyvips) `dzsave` builds the whole image pyramid in one pass — fast and
low-memory even for 100 MP inputs. Output layout (Deep Zoom, 'dz'):

    <TILES_DIR>/<id>/img.dzi
    <TILES_DIR>/<id>/img_files/<level>/<col>_<row>.jpg

OpenSeadragon, given the `.dzi` URL, derives every tile URL from the sibling
`img_files/` folder automatically.
"""
import logging
import shutil
from pathlib import Path

import pyvips

from app import config

logger = logging.getLogger(__name__)

TILE_SIZE = 256
OVERLAP = 1
JPEG_Q = 80


def tiles_dir(sample_id: str) -> Path:
    return config.TILES_DIR / sample_id


def dzi_path(sample_id: str) -> Path:
    return tiles_dir(sample_id) / "img.dzi"


def has_tiles(sample_id: str) -> bool:
    return dzi_path(sample_id).is_file()


def generate_tiles(image_path: Path, sample_id: str) -> bool:
    """Build the DZI pyramid for one image. Returns True on success.

    Idempotent: an existing tile set for this id is rebuilt. On any failure the
    partial output is removed and False is returned (caller falls back to the
    plain downscaled <img>).
    """
    out = tiles_dir(sample_id)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    try:
        img = pyvips.Image.new_from_file(str(image_path), access="sequential")
        if img.hasalpha():
            img = img.flatten(background=[255, 255, 255])
        # dzsave writes <out>/img.dzi + <out>/img_files/<level>/<col>_<row>.jpg
        img.dzsave(str(out / "img"), suffix=f".jpg[Q={JPEG_Q}]",
                   tile_size=TILE_SIZE, overlap=OVERLAP, layout="dz")
        ok = dzi_path(sample_id).is_file()
        if not ok:
            shutil.rmtree(out, ignore_errors=True)
        return ok
    except Exception as exc:  # noqa: BLE001 — tiling is best-effort
        logger.warning("tiling failed for %s: %s", sample_id, exc)
        shutil.rmtree(out, ignore_errors=True)
        return False
