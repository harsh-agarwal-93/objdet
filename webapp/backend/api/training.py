"""Training API endpoints."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from backend.models.training import (
    TaskStatusResponse,
    TrainingConfig,
    TrainingJobResponse,
)
from backend.services import celery_service

router = APIRouter()


@router.post("/submit", response_model=TrainingJobResponse)
def submit_training_job(config: TrainingConfig) -> TrainingJobResponse:
    """Submit a new training job to Celery.

    Args:
        config: Training configuration.

    Returns:
        Training job response with task ID.

    Raises:
        HTTPException: If job submission fails.
    """
    try:
        # Create output directory
        output_dir = config.output_dir or f"/tmp/objdet_training/{config.name}"
        os.makedirs(output_dir, exist_ok=True)

        # For now, use a dummy config path (in production, generate from config model)
        config_path = config.config_path or "/tmp/default_config.yaml"

        # Submit to Celery
        task_id = celery_service.submit_training_job(
            config_path=config_path,
            output_dir=output_dir,
            max_epochs=config.epochs,
            devices=1 if config.gpu == "auto" else 1,
            accelerator="auto" if config.gpu == "auto" else "gpu",
        )

        return TrainingJobResponse(
            task_id=task_id,
            status="PENDING",
            created_at=datetime.now(UTC).isoformat(),
            estimated_duration=f"{config.epochs * 2}min",  # Rough estimate
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit training job: {e!s}")


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_training_status(task_id: str) -> TaskStatusResponse:
    """Get status of a training task.

    Args:
        task_id: Celery task ID.

    Returns:
        Task status information.
    """
    try:
        status = celery_service.get_task_status(task_id)
        return TaskStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {e!s}")


@router.post("/cancel/{task_id}")
def cancel_training_job(task_id: str) -> dict[str, str]:
    """Cancel a running training job.

    Args:
        task_id: Celery task ID.

    Returns:
        Cancellation confirmation.
    """
    try:
        celery_service.cancel_task(task_id)
        return {"status": "cancelled", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {e!s}")


@router.get("/active")
def list_active_training_jobs() -> dict[str, list[dict]]:
    """List all active training jobs.

    Returns:
        List of active task information.
    """
    try:
        tasks = celery_service.list_active_tasks()
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list active tasks: {e!s}")
