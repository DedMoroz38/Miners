"""Typed, immutable configuration loaded from conf/default.yaml via OmegaConf.

`load_config()` merges the YAML with optional dotlist overrides (e.g.
["model.track=segformer_b3", "train.batch=16"]) and freezes the result into
nested frozen dataclasses. Paths are resolved to absolute against the module
location so scripts work from any cwd.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

# talc_quant/src/talc_quant/config.py -> talc_quant/
_PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = _PKG_ROOT / "conf" / "default.yaml"
# ML/ directory (miners/ML) — the anchor for relative data paths in the YAML.
_ML_DIR = _PKG_ROOT.parent


@dataclass(frozen=True)
class Paths:
    yolo_seg: Path
    part1: Path
    part2: Path
    panoramas: Path
    talc_zones: Path
    build_dir: Path
    runs_dir: Path
    sulfide_preprocess: Path


@dataclass(frozen=True)
class DataCfg:
    crop: int
    patches_per_epoch: int
    val_patches: int
    copy_paste_p: float
    fda_p: float
    fda_beta: float
    gray_drop_p: float
    ab_jitter: int
    rare_center_p: float
    gmm_conf: float
    gmm_max_pixels: int
    sulfide_min_l: int
    min_talc_polygon_px: int
    neg_per_folder: int


@dataclass(frozen=True)
class ScaleCfg:
    target_mag: float
    panorama_mag: float
    jitter: float


@dataclass(frozen=True)
class ModelCfg:
    track: str
    n_classes: int
    pretrained: bool
    tiny: bool


@dataclass(frozen=True)
class LossCfg:
    ce_w: float
    dice_w: float
    tversky_w: float
    area_w: float
    tversky_alpha: float
    tversky_beta: float
    tversky_gamma: float
    talc_class_weight: float


@dataclass(frozen=True)
class TrainCfg:
    folds: int
    seed: int
    epochs: int
    batch: int
    lr: float
    weight_decay: float
    warmup_frac: float
    patience: int
    amp: bool
    workers: int
    device: str


@dataclass(frozen=True)
class InferCfg:
    tile: int
    overlap: float
    tta_flips: bool
    tta_scales: tuple[float, ...]
    batch_tiles: int
    seg_prob_threshold: float
    talc_line: float


@dataclass(frozen=True)
class CalibrateCfg:
    method: str
    temperature_init: float
    flag_band: tuple[float, float]


@dataclass(frozen=True)
class ReportCfg:
    overlay_alpha: float
    talc_bgr: tuple[int, int, int]
    grid_cells: int


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: DataCfg
    scale: ScaleCfg
    model: ModelCfg
    loss: LossCfg
    train: TrainCfg
    infer: InferCfg
    calibrate: CalibrateCfg
    report: ReportCfg


def _resolve(p: str) -> Path:
    """Resolve a YAML path string to absolute, anchored at ML/ if relative."""
    path = Path(p)
    return path if path.is_absolute() else (_ML_DIR / path).resolve()


def _build_paths(node) -> Paths:
    return Paths(
        yolo_seg=_resolve(node.yolo_seg),
        part1=_resolve(node.part1),
        part2=_resolve(node.part2),
        panoramas=_resolve(node.panoramas),
        talc_zones=_resolve(node.talc_zones),
        build_dir=_resolve(node.build_dir),
        runs_dir=_resolve(node.runs_dir),
        sulfide_preprocess=_resolve(node.sulfide_preprocess),
    )


def load_config(
    yaml_path: Path | str = DEFAULT_YAML,
    overrides: Sequence[str] | None = None,
) -> Config:
    """Load YAML + dotlist overrides -> frozen Config."""
    cfg = OmegaConf.load(str(yaml_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    c = OmegaConf.to_container(cfg, resolve=True)
    return Config(
        paths=_build_paths(cfg.paths),
        data=DataCfg(**c["data"]),
        scale=ScaleCfg(**c["scale"]),
        model=ModelCfg(**c["model"]),
        loss=LossCfg(**c["loss"]),
        train=TrainCfg(**c["train"]),
        infer=InferCfg(
            tile=c["infer"]["tile"],
            overlap=c["infer"]["overlap"],
            tta_flips=c["infer"]["tta_flips"],
            tta_scales=tuple(c["infer"]["tta_scales"]),
            batch_tiles=c["infer"]["batch_tiles"],
            seg_prob_threshold=c["infer"]["seg_prob_threshold"],
            talc_line=c["infer"]["talc_line"],
        ),
        calibrate=CalibrateCfg(
            method=c["calibrate"]["method"],
            temperature_init=c["calibrate"]["temperature_init"],
            flag_band=tuple(c["calibrate"]["flag_band"]),
        ),
        report=ReportCfg(
            overlay_alpha=c["report"]["overlay_alpha"],
            talc_bgr=tuple(c["report"]["talc_bgr"]),
            grid_cells=c["report"]["grid_cells"],
        ),
    )
