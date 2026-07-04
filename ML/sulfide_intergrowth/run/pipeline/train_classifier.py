"""Step 4: ConvNeXt tile classifier (normal vs fine intergrowths).

Single grouped train/val split. Saves weights/classifier_best.pt +
weights/decision.json (threshold, F1s).
"""
import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

import _bootstrap  # noqa: F401
from src.trainer_module import ClassifierTrainer
from src.utils import get_device, resolve_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the classifier and report val / holdout image-level F1."""
    setup_logging()
    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))
    derived = resolve_path(cfg.paths.derived)
    weights = resolve_path(cfg.paths.weights)
    images = pd.read_csv(derived / "index_cached.csv").set_index("image_id")
    tiles = pd.read_csv(derived / "tiles.csv")
    logger.info("Classifier: %d images / %d tiles (device=%s)", len(images), len(tiles), device)
    trainer = ClassifierTrainer(cfg, device)
    decision = trainer.fit(images, tiles, derived / "cache", weights)
    logger.info("RESULT: VAL F1=%.4f | HOLDOUT F1=%.4f (target >= 0.90)",
                decision["val_f1"], decision["holdout_f1"])


if __name__ == "__main__":
    main()
