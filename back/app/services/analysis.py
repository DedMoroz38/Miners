"""Ore analysis: run the two ML pipelines + merge, return the frontend payload.

Orchestrates three subprocesses (each in its own venv — see app.config):
  1. talc      : ML/fusion/infer_one.py         -> talc.json + talc_mask.png
  2. sulfide   : sulfide_intergrowth infer_panorama.py -> <stem>_classmap.png (+ segments.csv)
  3. merge     : ML/merge/merge_phases.py        -> analysis.json (green/red/blue phase map,
                 metrics, verdict, normalized polygon segments)

The steps run sequentially in a per-job temp dir. Talc and sulfide are independent
and could be parallelized later; kept sequential for a simpler MVP.
"""
import json
import subprocess
import tempfile
from pathlib import Path

from app import config


def _run(cmd: list[str], cwd: str | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                          timeout=config.ML_STEP_TIMEOUT)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"{Path(cmd[1]).name} failed (exit {proc.returncode}):\n{tail}")


def analyze_sample(image_path: Path) -> dict:
    """Run talc + sulfide + merge; return the merged analysis dict (AnalysisResult)."""
    image_path = Path(image_path).resolve()
    stem = image_path.stem
    with tempfile.TemporaryDirectory(dir=str(config.JOBS_DIR)) as tmp:
        tmp = Path(tmp)
        talc_dir = tmp / "talc"
        sulfide_out = tmp / "sulfide"

        # 1. talc (classifier-gated YOLO-seg) -> talc.json + talc_mask.png
        _run([config.TALC_PYTHON, config.TALC_SCRIPT,
              "--image", str(image_path), "--out", str(talc_dir)])

        # 2. sulfide (U-Net + tile classifier) -> reports/<stem>_classmap.png
        _run([config.SULFIDE_PYTHON, config.SULFIDE_SCRIPT,
              f"+input={image_path}",
              f"paths.weights={config.SULFIDE_WEIGHTS_DIR}",
              f"paths.outputs={sulfide_out}"],
             cwd=config.SULFIDE_CWD)
        reports = sulfide_out / "reports"
        classmap = reports / f"{stem}_classmap.png"
        segments_csv = reports / f"{stem}_segments.csv"
        decision_json = Path(config.SULFIDE_WEIGHTS_DIR) / "decision.json"

        # 3. merge -> analysis.json (overlap rule: sulfide overrides talc)
        out_json = tmp / "analysis.json"
        _run([config.MERGE_PYTHON, config.MERGE_SCRIPT,
              "--sulfide-classmap", str(classmap),
              "--talc-mask", str(talc_dir / "talc_mask.png"),
              "--talc-json", str(talc_dir / "talc.json"),
              "--sulfide-segments", str(segments_csv),
              "--decision-json", str(decision_json),
              "--out", str(out_json)])

        return json.loads(out_json.read_text(encoding="utf-8"))
