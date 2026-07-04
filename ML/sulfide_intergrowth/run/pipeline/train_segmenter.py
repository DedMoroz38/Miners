"""Step 3: train the sulfide U-Net on pseudo-labelled tiles; saves weights/segmenter_best.pt."""
import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

import _bootstrap  # noqa: F401
from src.trainer_module import SegTrainer
from src.utils import get_device, resolve_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the segmenter on trainval images (holdout untouched)."""
    setup_logging()
    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))
    derived = resolve_path(cfg.paths.derived)
    weights = resolve_path(cfg.paths.weights)
    df = pd.read_csv(derived / "index_cached.csv")
    frame = df[df["split"] == "trainval"].reset_index(drop=True)
    logger.info("Training segmenter on %d images (device=%s)", len(frame), device)
    trainer = SegTrainer(cfg, device)
    trainer.fit(frame, derived / "cache", weights)


if __name__ == "__main__":
    main()
