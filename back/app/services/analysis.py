"""Ore analysis: run the ML pipeline(s) + merge, return the frontend payload.

Orchestrates subprocesses (each in its own python — see app.config):
  1. talc      : ML/talc_infer/infer.py          -> talc.json + talc_mask.png
  2. sulfide   : sulfide_intergrowth infer_panorama.py -> <stem>_classmap.png (+ segments.csv)
                 — OPTIONAL: runs only when its weights are present
                 (config.sulfide_available()); otherwise we run talc-only.
  3. merge     : ML/merge/merge_phases.py         -> analysis.json (green/red/blue phase
                 map, metrics, verdict, normalized polygon segments)

Talc-only mode: the sulfide segmenter has no weights yet, so we synthesize an
all-background sulfide class map at the talc-mask resolution and feed that to the
merge step. The merge then produces a pure-talc (blue) result — the overlap rule
still holds (sulfide would override talc, but there is no sulfide here).

The steps run sequentially in a per-job temp dir.
"""
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from app import config


def _run(cmd: list[str], cwd: str | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                          timeout=config.ML_STEP_TIMEOUT)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"{Path(cmd[1]).name} failed (exit {proc.returncode}):\n{tail}")


def analyze_sample(image_path: Path) -> dict:
    """Run talc (+ sulfide if available) + merge; return the AnalysisResult dict."""
    image_path = Path(image_path).resolve()
    stem = image_path.stem
    with tempfile.TemporaryDirectory(dir=str(config.JOBS_DIR)) as tmp:
        tmp = Path(tmp)
        talc_dir = tmp / "talc"

        # 1. talc (classifier-gated YOLO-seg fusion) -> talc.json + talc_mask.png
        # Панорамы talc_infer тайлит сам (авто по размеру); связка
        # профиль<->веса тоже у него (--profile auto).
        talc_cmd = [config.TALC_PYTHON, config.TALC_SCRIPT,
                    "--image", str(image_path), "--out", str(talc_dir)]
        if config.TALC_SEG_WEIGHTS:
            talc_cmd += ["--seg-weights", config.TALC_SEG_WEIGHTS]
        _run(talc_cmd)
        talc_mask = talc_dir / "talc_mask.png"
        talc_json = talc_dir / "talc.json"

        # 2. sulfide (optional — only when weights exist)
        segments_csv: Path | None = None
        decision_json: Path | None = None
        if config.sulfide_available():
            sulfide_out = tmp / "sulfide"
            _run([config.SULFIDE_PYTHON, config.SULFIDE_SCRIPT,
                  f"+input={image_path}",
                  f"paths.weights={config.SULFIDE_WEIGHTS_DIR}",
                  f"paths.outputs={sulfide_out}"],
                 cwd=config.SULFIDE_CWD)
            reports = sulfide_out / "reports"
            classmap = reports / f"{stem}_classmap.png"
            segments_csv = reports / f"{stem}_segments.csv"
            decision_json = Path(config.SULFIDE_WEIGHTS_DIR) / "decision.json"
        else:
            # talc-only: all-background sulfide map at the talc-mask resolution.
            classmap = tmp / "sulfide_empty_classmap.png"
            with Image.open(talc_mask) as tm:
                w, h = tm.size
            Image.new("L", (w, h), 0).save(classmap)

        # 3. merge -> analysis.json (overlap rule: sulfide overrides talc)
        out_json = tmp / "analysis.json"
        cmd = [config.MERGE_PYTHON, config.MERGE_SCRIPT,
               "--sulfide-classmap", str(classmap),
               "--talc-mask", str(talc_mask),
               "--talc-json", str(talc_json),
               "--out", str(out_json)]
        if segments_csv is not None:
            cmd += ["--sulfide-segments", str(segments_csv)]
        if decision_json is not None:
            cmd += ["--decision-json", str(decision_json)]
        _run(cmd)

        return json.loads(out_json.read_text(encoding="utf-8"))
