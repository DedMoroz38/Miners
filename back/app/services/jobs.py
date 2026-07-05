"""In-memory async job store for analysis runs.

MVP: jobs live in a process-local dict and run on a small thread pool. Not
persisted and not shared across workers — run uvicorn single-process for now, or
swap this for a real queue later. The API contract (create -> poll) stays the same.
"""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_executor = ThreadPoolExecutor(max_workers=2)
_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def create_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    return job_id


def submit(job_id: str, fn: Callable[..., dict], *args: Any) -> None:
    """Run `fn(*args)` in the background; store its dict result or the error."""
    def run() -> None:
        with _lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "running"
        try:
            result = fn(*args)
            with _lock:
                _jobs[job_id].update(status="done", result=result)
        except Exception as exc:  # surface any failure to the poller
            with _lock:
                _jobs[job_id].update(status="error", error=str(exc))

    _executor.submit(run)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None
