"""Step 5: full inference on panoramas -> overlay + metrics table.

Usage:
    python infer_panorama.py                          # all panoramas from data
    python infer_panorama.py +input=/path/to/img.jpg  # a single image
"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

import _bootstrap  # noqa: F401
from src.inference import PanoramaPipeline
from src.utils import get_device, resolve_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the saved-weights pipeline on one or all panoramas."""
    setup_logging()
    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))
    weights = resolve_path(cfg.paths.weights)
    out_dir = resolve_path(cfg.paths.outputs) / "reports"
    pipeline = PanoramaPipeline(cfg, weights, device)

    if "input" in cfg and cfg.input:
        targets = [Path(str(cfg.input))]
    else:
        pano_dir = resolve_path(cfg.paths.data_root) / str(cfg.data.panoramas_dir)
        targets = sorted(p for p in pano_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".bmp"})
    if not targets:
        raise FileNotFoundError("No input panoramas found")
    for path in targets:
        _, table = pipeline.run(path, out_dir)
        logger.info("%s:\n%s", path.name, table.to_string(index=False))


if __name__ == "__main__":
    main()
