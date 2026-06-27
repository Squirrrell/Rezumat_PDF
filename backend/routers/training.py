"""Offline T5 fine-tuning: start jobs, poll progress, run ROUGE evaluation.

This drives the standalone experiment under ``training/`` from the API. Jobs
run in a background thread and are tracked in-memory, so they are lost on
backend restart and only one may be active at a time.
"""

from __future__ import annotations

import importlib.util
import logging
import threading

from fastapi import APIRouter, HTTPException

from backend import training_store
from backend.schemas import (
    TrainingEvaluateRequest,
    TrainingInfoResponse,
    TrainingJobResponse,
    TrainingStartRequest,
)
from src.device_utils import cuda_device_info, get_device

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/training", tags=["training"])

_TRAINING_PACKAGES = ("datasets", "rouge_score")


def _check_dependencies() -> tuple[bool, str | None]:
    missing = [pkg for pkg in _TRAINING_PACKAGES if importlib.util.find_spec(pkg) is None]
    if missing:
        return False, (
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -r training/requirements-training.txt"
        )
    return True, None


def _job_to_response(job: training_store.TrainingJob) -> TrainingJobResponse:
    return TrainingJobResponse(
        job_id=job.job_id,
        status=job.status,
        config=job.config,
        total_epochs=job.total_epochs,
        current_epoch=job.current_epoch,
        current_step=job.current_step,
        log_history=job.log_history,
        checkpoint_path=job.checkpoint_path,
        error=job.error,
    )


def _run_job(job_id: str, config) -> None:
    from training import service

    training_store.update_job(job_id, status="running")

    def on_progress(update: dict) -> None:
        training_store.update_job(
            job_id,
            current_epoch=update.get("epoch") or 0.0,
            current_step=update.get("step") or 0,
            log_history=update.get("log_history") or [],
        )

    def should_cancel() -> bool:
        try:
            return training_store.get_job(job_id).cancel_requested
        except KeyError:
            return True

    try:
        result = service.run_training(config, on_progress=on_progress, should_cancel=should_cancel)
    except Exception as exc:  # noqa: BLE001 - surface any training failure to the UI
        logger.exception("Training job %s failed", job_id)
        training_store.update_job(job_id, status="failed", error=str(exc))
        return

    job = training_store.get_job(job_id)
    final_status = "cancelled" if job.cancel_requested else "completed"
    training_store.update_job(
        job_id,
        status=final_status,
        checkpoint_path=result.get("checkpoint_path"),
        log_history=result.get("log_history") or job.log_history,
    )


@router.get("/info", response_model=TrainingInfoResponse)
def training_info() -> TrainingInfoResponse:
    from training import service

    deps_ok, deps_message = _check_dependencies()
    cuda_info = cuda_device_info()
    active = training_store.get_active_job()

    return TrainingInfoResponse(
        device=get_device(),
        cuda_available=cuda_info["cuda_available"],
        gpu_name=cuda_info.get("gpu_name"),
        dependencies_installed=deps_ok,
        dependencies_message=deps_message,
        default_dataset_config=service.DEFAULT_DATASET_CONFIG,
        default_epochs=service.TrainingConfig().epochs,
        checkpoints=service.list_checkpoints(),
        final_checkpoint=service.find_final_checkpoint(),
        last_results=service.load_last_results(),
        active_job=_job_to_response(active) if active else None,
    )


@router.post("/start", response_model=TrainingJobResponse)
def start_training(body: TrainingStartRequest) -> TrainingJobResponse:
    from training import service

    deps_ok, deps_message = _check_dependencies()
    if not deps_ok:
        raise HTTPException(status_code=400, detail=deps_message)

    config = service.TrainingConfig(
        model_name=body.model_name,
        dataset_config=body.dataset_config,
        train_size=body.train_size,
        val_size=body.val_size,
        epochs=body.epochs,
        learning_rate=body.learning_rate,
    )

    try:
        job = training_store.create_job(config.to_dict(), total_epochs=config.epochs)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    thread = threading.Thread(target=_run_job, args=(job.job_id, config), daemon=True)
    thread.start()

    return _job_to_response(job)


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_job(job_id: str) -> TrainingJobResponse:
    try:
        job = training_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _job_to_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
def cancel_job(job_id: str) -> TrainingJobResponse:
    try:
        job = training_store.request_cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _job_to_response(job)


@router.post("/evaluate")
def evaluate(body: TrainingEvaluateRequest) -> dict:
    from training import service

    deps_ok, deps_message = _check_dependencies()
    if not deps_ok:
        raise HTTPException(status_code=400, detail=deps_message)

    if training_store.get_active_job() is not None:
        raise HTTPException(
            status_code=409,
            detail="A training job is still active. Wait for it to finish before evaluating.",
        )

    checkpoint = body.checkpoint or (service.find_final_checkpoint() or "")
    if body.run_finetuned and not checkpoint:
        raise HTTPException(
            status_code=400,
            detail="No fine-tuned checkpoint found. Train a model first or pass a checkpoint path.",
        )

    try:
        results = service.run_rouge_evaluation(
            dataset_config=body.dataset_config,
            checkpoint=checkpoint,
            train_size=body.train_size,
            eval_size=body.eval_size,
            run_baseline=body.run_baseline,
            run_finetuned=body.run_finetuned,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface eval failures to the UI
        logger.exception("ROUGE evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return results
