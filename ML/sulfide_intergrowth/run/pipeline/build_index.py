"""Step 1: scan data folders -> index.csv (labels, slide groups, holdout split)."""
import logging

import hydra
from omegaconf import DictConfig

import _bootstrap  # noqa: F401
from src.data_module.indexing import build_index, save_index
from src.utils import resolve_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build and persist the image index."""
    setup_logging()
    set_seed(int(cfg.seed))
    data_root = resolve_path(cfg.paths.data_root)
    derived = resolve_path(cfg.paths.derived)
    df = build_index(
        data_root=data_root,
        part_dirs=list(cfg.data.part_dirs),
        class_dirs={k: list(v) for k, v in cfg.data.class_dirs.items()},
        ignore_dirs=list(cfg.data.ignore_dirs),
        holdout_fraction=float(cfg.data.holdout_fraction),
        seed=int(cfg.seed),
        limit_per_class=(int(cfg.data.limit_per_class)
                         if cfg.data.limit_per_class is not None else None),
    )
    save_index(df, derived / "index.csv")


if __name__ == "__main__":
    main()
