"""Tests for the training job store and API router (no real training)."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import training_store
from backend.routers import training as training_router
from training import service


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setenv("ML_ALLOW_CPU", "1")
    training_store._jobs.clear()
    yield
    training_store._jobs.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(training_router.router)
    return TestClient(app)


def test_single_active_job_guard():
    training_store.create_job({"epochs": 1}, total_epochs=1)
    with pytest.raises(RuntimeError):
        training_store.create_job({"epochs": 1}, total_epochs=1)


def test_completed_job_frees_slot():
    job = training_store.create_job({}, total_epochs=1)
    training_store.update_job(job.job_id, status="completed")
    second = training_store.create_job({}, total_epochs=1)
    assert second.job_id != job.job_id


def test_request_cancel_sets_flag():
    job = training_store.create_job({}, total_epochs=1)
    updated = training_store.request_cancel(job.job_id)
    assert updated.cancel_requested is True


def test_run_job_success(monkeypatch):
    job = training_store.create_job({}, total_epochs=2)

    def fake_run(config, on_progress=None, should_cancel=None):
        on_progress(
            {"epoch": 1.0, "step": 7, "log_history": [{"epoch": 1.0, "eval_loss": 2.0}]}
        )
        return {
            "checkpoint_path": "/tmp/final",
            "log_history": [{"epoch": 1.0, "eval_loss": 2.0}],
            "device": "cpu",
        }

    monkeypatch.setattr(service, "run_training", fake_run)
    training_router._run_job(job.job_id, object())

    done = training_store.get_job(job.job_id)
    assert done.status == "completed"
    assert done.checkpoint_path == "/tmp/final"
    assert done.current_step == 7


def test_run_job_failure(monkeypatch):
    job = training_store.create_job({}, total_epochs=1)

    def fake_run(config, on_progress=None, should_cancel=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "run_training", fake_run)
    training_router._run_job(job.job_id, object())

    done = training_store.get_job(job.job_id)
    assert done.status == "failed"
    assert "boom" in done.error


def test_run_job_cancelled(monkeypatch):
    job = training_store.create_job({}, total_epochs=1)
    training_store.request_cancel(job.job_id)

    def fake_run(config, on_progress=None, should_cancel=None):
        assert should_cancel() is True
        return {"checkpoint_path": None, "log_history": []}

    monkeypatch.setattr(service, "run_training", fake_run)
    training_router._run_job(job.job_id, object())

    done = training_store.get_job(job.job_id)
    assert done.status == "cancelled"


def test_info_endpoint(client):
    resp = client.get("/api/training/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_dataset_config"] == "arxiv"
    assert data["default_epochs"] == 5
    assert "dependencies_installed" in data


def test_start_blocked_without_dependencies(client, monkeypatch):
    monkeypatch.setattr(training_router, "_check_dependencies", lambda: (False, "missing deps"))
    resp = client.post("/api/training/start", json={"epochs": 1, "train_size": 10, "val_size": 5})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "missing deps"


def test_evaluate_blocked_without_dependencies(client, monkeypatch):
    monkeypatch.setattr(training_router, "_check_dependencies", lambda: (False, "missing deps"))
    resp = client.post("/api/training/evaluate", json={})
    assert resp.status_code == 400


def test_start_and_poll_until_complete(client, monkeypatch):
    monkeypatch.setattr(training_router, "_check_dependencies", lambda: (True, None))

    def fake_run(config, on_progress=None, should_cancel=None):
        return {"checkpoint_path": "/tmp/final", "log_history": [], "device": "cpu"}

    monkeypatch.setattr(service, "run_training", fake_run)

    resp = client.post(
        "/api/training/start", json={"epochs": 1, "train_size": 10, "val_size": 5}
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    status = "queued"
    for _ in range(100):
        status = client.get(f"/api/training/jobs/{job_id}").json()["status"]
        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.05)

    assert status == "completed"
