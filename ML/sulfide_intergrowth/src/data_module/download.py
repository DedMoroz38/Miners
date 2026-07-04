"""One-time dataset provisioning from Google Drive.

`ensure_dataset` is idempotent: it downloads into `data_root` only when the
expected class folders are missing or empty, so every run after the first
trains directly off the local copy. Supports both a zipped dataset and a shared
Drive folder via `gdown`.
"""
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceConfig:
    gdrive_url: str | None = None
    kind: str = "auto"              # auto | zip | folder
    strip_top_level: bool = True
    force: bool = False

    @staticmethod
    def from_cfg(cfg) -> "SourceConfig":
        """Build from an OmegaConf node (cfg.data.source), tolerating absence."""
        url = getattr(cfg, "gdrive_url", None)
        return SourceConfig(
            gdrive_url=str(url) if url else None,
            kind=str(getattr(cfg, "kind", "auto")),
            strip_top_level=bool(getattr(cfg, "strip_top_level", True)),
            force=bool(getattr(cfg, "force", False)),
        )


def _is_populated(data_root: Path, expected_subdirs: list[str]) -> bool:
    """True if at least one expected class folder exists and holds files."""
    for name in expected_subdirs:
        d = data_root / name
        if d.is_dir() and any(d.rglob("*")):
            return True
    return False


def _resolve_kind(url: str, kind: str) -> str:
    if kind != "auto":
        return kind
    if "/folders/" in url:
        return "folder"
    return "zip"


def _flatten_single_top_dir(root: Path) -> None:
    """If everything extracted under one wrapper dir, lift it into `root`."""
    entries = [p for p in root.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        for item in list(wrapper.iterdir()):
            shutil.move(str(item), str(root / item.name))
        wrapper.rmdir()
        logger.info("Flattened wrapper directory: %s", wrapper.name)


def _download_zip(url: str, data_root: Path, strip_top_level: bool) -> None:
    import gdown

    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "dataset.zip"
        out = gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)
        if out is None or not zip_path.is_file():
            raise RuntimeError(
                f"gdown failed to download {url}. Check the link is shared "
                "'Anyone with the link' and points to a file.")
        logger.info("Extracting %s ...", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_root)
    if strip_top_level:
        _flatten_single_top_dir(data_root)


def _download_folder(url: str, data_root: Path) -> None:
    import gdown

    data_root.mkdir(parents=True, exist_ok=True)
    result = gdown.download_folder(url=url, output=str(data_root), quiet=False,
                                   use_cookies=False)
    if not result:
        raise RuntimeError(
            f"gdown failed to download folder {url}. Check sharing permissions.")


def ensure_dataset(data_root: Path, expected_subdirs: list[str],
                   source: SourceConfig) -> Path:
    """Guarantee the dataset is present under `data_root`, downloading once.

    Args:
        data_root: local destination (also where training reads from).
        expected_subdirs: class part-dirs used for the idempotency check.
        source: Google Drive source configuration.

    Returns:
        The `data_root` path.

    Raises:
        FileNotFoundError: data absent and no `gdrive_url` configured.
        RuntimeError: download failed.
    """
    data_root = Path(data_root)
    if not source.force and _is_populated(data_root, expected_subdirs):
        logger.info("Dataset already present at %s — skipping download", data_root)
        return data_root
    if not source.gdrive_url:
        raise FileNotFoundError(
            f"Dataset not found under {data_root} and data.source.gdrive_url is "
            "unset. Set it (share link to a .zip or Drive folder) or place the "
            "data manually.")
    kind = _resolve_kind(source.gdrive_url, source.kind)
    logger.info("Fetching dataset (%s) from %s -> %s", kind, source.gdrive_url, data_root)
    if kind == "folder":
        _download_folder(source.gdrive_url, data_root)
    else:
        _download_zip(source.gdrive_url, data_root, source.strip_top_level)
    if not _is_populated(data_root, expected_subdirs):
        raise RuntimeError(
            f"Download finished but expected folders {expected_subdirs} are still "
            f"missing under {data_root}. Check the archive layout / "
            "data.source.strip_top_level.")
    logger.info("Dataset ready at %s", data_root)
    return data_root
