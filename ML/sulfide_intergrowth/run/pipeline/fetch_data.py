"""Step 0 (optional): download the dataset from Google Drive into data_root.

Idempotent — running it again is a no-op once the data is present. build_index.py
calls the same logic automatically, so this standalone step is only needed if
you want to fetch the data ahead of time (or with data.source.force=true).

Usage:
    python fetch_data.py data.source.gdrive_url=https://drive.google.com/...
    python fetch_data.py data.source.gdrive_url=<url> data.source.force=true
"""
import logging

import hydra
from omegaconf import DictConfig

import _bootstrap  # noqa: F401
from src.data_module.download import SourceConfig, ensure_dataset
from src.utils import resolve_path, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Fetch the dataset once (skips if already present)."""
    setup_logging()
    data_root = resolve_path(cfg.paths.data_root)
    ensure_dataset(data_root, list(cfg.data.part_dirs), SourceConfig.from_cfg(cfg.data.source))
    logger.info("Data root ready: %s", data_root)


if __name__ == "__main__":
    main()
