"""In-memory store for offline T5 fine-tuning jobs.

Mirrors the lightweight pattern used by ``session_store``. Only one job may be
active at a time to avoid exhausting GPU memory with concurrent runs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
_ACTIVE_STATUSES = {"queued", "running"}


@dataclass
class TrainingJob:
    job_id: str
    status: JobStatus
    config: dict[str, Any]
    total_epochs: int
    current_epoch: float = 0.0
    current_step: int = 0
    log_history: list[dict] = field(default_factory=list)
    checkpoint_path: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_jobs: dict[str, TrainingJob] = {}
_lock = threading.Lock()


def get_active_job() -> TrainingJob | None:
    for job in _jobs.values():
        if job.status in _ACTIVE_STATUSES:
            return job
    return None


def create_job(config: dict[str, Any], total_epochs: int) -> TrainingJob:
    """Create a queued job; raises RuntimeError if one is already active."""
    with _lock:
        active = get_active_job()
        if active is not None:
            raise RuntimeError(
                f"A training job is already {active.status} (id {active.job_id}). "
                "Wait for it to finish or cancel it first."
            )
        job = TrainingJob(
            job_id=str(uuid4()),
            status="queued",
            config=config,
            total_epochs=total_epochs,
        )
        _jobs[job.job_id] = job
        return job


def get_job(job_id: str) -> TrainingJob:
    job = _jobs.get(job_id)
    if job is None:
        raise KeyError(f"Training job not found: {job_id}")
    return job


def update_job(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def request_cancel(job_id: str) -> TrainingJob:
    with _lock:
        job = get_job(job_id)
        if job.status in _ACTIVE_STATUSES:
            job.cancel_requested = True
        return job
