"""End-to-end API contract test (no ML weights needed).

Drives the real FastAPI app with an in-memory image and a monkeypatched
`analyze_sample` that returns a genuine merge_phases payload, verifying:
  upload -> analyze (202 + job_id) -> poll -> done + AnalysisResultOut result,
and that the merge output validates against the pydantic response schema the
frontend consumes.

Run:  cd back && python test_api_contract.py     (needs fastapi/httpx/pillow + numpy/cv2)
"""
import io
import os
import sys
import time
from pathlib import Path

# import the real merge to produce a realistic payload
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ML" / "merge"))
import numpy as np
import merge_phases as M  # noqa: E402
from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import app` works
from app.main import app  # noqa: E402
from app.api import samples as samples_api  # noqa: E402
from app.schemas import AnalysisResultOut  # noqa: E402

H, W = 120, 160


def real_merge_payload(_image_path):
    """Produce a payload from the actual merge code (sulfide + overlapping talc)."""
    cm = np.zeros((H, W), np.uint8)
    cm[20:50, 20:70] = 1     # common
    cm[70:100, 100:140] = 2  # thin
    talc = np.zeros((H, W), np.uint8)
    talc[30:60, 60:110] = 255  # overlaps common + spills into bg
    payload, _ = M.build_payload(cm, talc, None, None, f1=0.42)
    return payload


def main():
    fails = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)

    # 1. The merge payload validates against the response schema (front contract).
    payload = real_merge_payload(None)
    model = AnalysisResultOut(**payload)
    check("merge payload validates as AnalysisResultOut", isinstance(model, AnalysisResultOut))
    check("has verdict + conclusion + segments",
          bool(model.verdict) and bool(model.conclusion) and len(model.segments) > 0)

    # 2. Monkeypatch the heavy pipeline; drive the API.
    samples_api.analyze_sample = real_merge_payload
    client = TestClient(app)

    check("health ok", client.get("/api/health").json().get("status") == "ok")

    # in-memory PNG upload
    buf = io.BytesIO()
    Image.new("RGB", (W, H), (30, 40, 50)).save(buf, format="PNG")
    buf.seek(0)
    up = client.post("/api/samples", files={"file": ("slide.png", buf, "image/png")})
    check("upload 201", up.status_code == 201)
    sample_id = up.json().get("id")
    check("upload returns id + image_url",
          bool(sample_id) and up.json().get("image_url", "").endswith("/image"))

    # start analysis
    an = client.post(f"/api/samples/{sample_id}/analyze")
    check("analyze 202", an.status_code == 202)
    job_id = an.json().get("job_id")
    check("analyze returns job_id", bool(job_id))

    # poll to completion
    result = None
    for _ in range(50):
        r = client.get(f"/api/samples/{sample_id}/analyze/{job_id}").json()
        if r["status"] in ("done", "error"):
            result = r
            break
        time.sleep(0.05)
    check("job reached done", result and result["status"] == "done")
    if result and result["status"] == "done":
        res = result["result"]
        check("result matches merge output (talcShare)",
              abs(res["talcShare"] - payload["talcShare"]) < 1e-9)
        check("result carries normalized polygons",
              all(0 <= x <= 1 and 0 <= y <= 1
                  for s in res["segments"] for ring in s["polygons"] for x, y in ring))
        check("sulfide overrides preserved through API",
              res["sulfideShare"] == payload["sulfideShare"])
    elif result:
        print("   error:", result.get("error"))

    # 3. 404s for unknown ids
    check("unknown sample analyze -> 404",
          client.post("/api/samples/deadbeef/analyze").status_code == 404)

    # cleanup the uploaded file
    for p in samples_api.config.STORAGE_DIR.glob(f"{sample_id}.*"):
        p.unlink(missing_ok=True)

    return fails


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    fails = main()
    print("\nRESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    sys.exit(1 if fails else 0)
