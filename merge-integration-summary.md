# Merge integration — verification & wiring summary

Continuation of the `started merging` commit (c78895d). Goal: finalize the merge
of all segmentation components (sulfide intergrowth + talc fusion) into one
coordinate system, wire ML → backend → frontend, and make the frontend able to
edit segments.

**Key outcome:** the `started merging` commit was already structurally complete and
correct. The remaining work turned out to be **verification + hardening + Docker
wiring**, not rewriting logic. No implementation code was changed — verification
proved it was right.

---

## What was verified

### The merge core (`ML/merge/merge_phases.py`) — the crux
Driven with synthetic inputs via `ML/merge/test_merge_phases.py` (numpy + opencv
only, no weights). Confirms the two overlap rules from the spec:

- **talc ↔ sulfide → sulfide wins** — talc only fills background pixels; never
  painted over a sulfide grain.
- **talc ↔ talc** — single union mask; pruned (`kept=False`) segments excluded
  from the confidence map.
- **resolution mismatch** — a ½-resolution talc mask still aligns (NEAREST
  resize); output emitted at the sulfide classmap resolution.
- metrics (`sulfideShare`, `commonShare`, `thinShare`, `talcShare`), verdict,
  `f1` passthrough, and **normalized polygons in [0,1]** all correct.

Degenerate inputs also covered: no talc, no sulfide (no divide-by-zero), talc
entirely inside a grain (→ 0 talc), fully talcose (→ `оталькованная`).
**Result: 27/27 assertions pass.**

### The backend API contract (`back/test_api_contract.py`)
`TestClient` drives the real FastAPI app with a genuine `merge_phases` payload
and a monkeypatched heavy pipeline:

- `upload (201)` → `analyze (202 + job_id)` → poll → `done` + result.
- The merge output validates against the `AnalysisResultOut` schema the frontend
  consumes.
- Normalized polygons and the sulfide-override survive the round trip.
- Unknown ids → `404`.

**Result: 12/12 assertions pass.**

### Data contracts (read-through, confirmed aligned)
- `infer_one.py` `seg_record` keys (`confidence`, `kept`, `polygons_norm`) ==
  what `merge_phases.build_talc_conf_map` reads.
- sulfide classmap encoding `0=bg / 1=обычные / 2=тонкие` == merge's
  `BG/COMMON/THIN`; talc added as `TALC` (blue) by merge.
- `merge_phases` payload == `AnalysisResultOut` (backend) == `AnalysisResult`
  (frontend `entities/analysis`).
- Verdict/conclusion rule identical in `merge_phases.classify` and
  `front/.../metrics.ts:classify`.

---

## What was added

| File | Purpose |
|---|---|
| `ML/merge/test_merge_phases.py` | Merge-core verification (overlaps, metrics, edge cases) — no weights |
| `back/test_api_contract.py` | API contract verification (upload→analyze→poll→result) — no weights |
| `back/Dockerfile` | Lightweight FastAPI image (API only; ML runs in mounted venvs) |
| `back/.dockerignore` | Exclude venv/caches/storage from the build context |
| `docker-compose.yaml` (root) | Wires back (:8000) + front (:3000); mounts `ML/`; sets ML env vars |
| `back/README.md` (edited) | Added `/analyze` endpoints, compose flow, degradation, test commands |

### Docker / deployment (Linux target)
The backend image serves the API only. The three ML pipelines run as subprocesses
in their own venvs (per `app/config.py`); baking multi-GB torch/ultralytics venvs
into the API image is impractical, so `docker-compose.yaml` mounts the repo's
`ML/` tree and points the backend at the pre-built Linux venvs + weights via env
vars. All 6 env-var names were verified to match `config.py` exactly.

**Graceful degradation (confirmed by design):** without venvs/weights the API
starts and uploads work; only `/analyze` errors — a missing venv makes
`subprocess.run` raise `FileNotFoundError`, caught in `jobs.py`, stored as
`status: "error"`, and surfaced to the frontend. The service stays up.

---

## Open finding (documented, not fixed)

At talc **9.5–9.99 %**, the verdict text renders **"тальк — 10 % (≤10 %)"** because
both `classify()` implementations format with `{:.0f}` (9.5 → "10"). The rule is
identical in `ML/merge/merge_phases.py` and `front/src/features/analyze-sample/lib/metrics.ts`
— a fix must touch **both** to keep the string contract in sync. Left as a
decision for the team since it changes user-facing strings.

---

## Genuinely blocked (not code)

True ML end-to-end needs the **trained weights + Linux venvs**, which aren't in
git (expected — large artifacts). Everything *around* the ML — merge logic,
backend orchestration, API contract, frontend rendering/editing — is verified
without them. See `ML/sulfide_intergrowth/weights/README.md` for the weight files
the sulfide step expects (`decision.json` mandatory).

---

## How to run the checks

```bash
python ML/merge/test_merge_phases.py     # merge rules + edge cases
cd back && python test_api_contract.py   # API upload→analyze→poll→result
```

Both need `numpy`, `opencv-python`, `fastapi`, `httpx`, `pillow`,
`python-multipart` (all in `back/requirements.txt`).
