"""Celery service for task submission and monitoring."""

from __future__ import annotations

import os
from typing import Any

from celery.result import AsyncResult

from backend.core.config import settings

# Access the existing Celery app from objdet
os.environ["CELERY_BROKER_URL"] = settings.celery_broker_url
os.environ["CELERY_RESULT_BACKEND"] = settings.celery_result_backend

from objdet.pipelines.celery_app import app as celery_app  # noqa: E402


def submit_training_job(
    config_path: str,
    output_dir: str,
    max_epochs: int | None = None,
    devices: int = 1,
    accelerator: str = "auto",
) -> str:
    """Submit a training job to Celery.

    Args:
        config_path: Path to training config YAML.
        output_dir: Output directory for checkpoints and logs.
        max_epochs: Override max epochs from config.
        devices: Number of GPU devices.
        accelerator: Accelerator type (auto, gpu, cpu).

    Returns:
        Celery task ID.
    """
    from objdet.pipelines.tasks import train_model

    result = train_model.delay(
        config_path=config_path,
        output_dir=output_dir,
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
    )

    return result.id


def get_task_status(task_id: str) -> dict[str, Any]:
    """Get status of a Celery task.

    Args:
        task_id: Celery task ID.

    Returns:
        Task status information.
    """
    result = AsyncResult(task_id, app=celery_app)

    response: dict[str, Any] = {
        "task_id": task_id,
        "status": result.state,
        "result": None,
        "error": None,
        "progress": None,
    }

    if result.state == "SUCCESS":
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.info)
    elif result.state == "PROGRESS":
        response["progress"] = result.info

    return response


def cancel_task(task_id: str) -> bool:
    """Cancel a running Celery task.

    Args:
        task_id: Celery task ID.

    Returns:
        True if cancellation request was sent.
    """
    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True)
    return True


def list_active_tasks() -> list[dict[str, Any]]:
    """List all active tasks.

    Returns:
        List of active task information.
    """
    # Get active tasks from Celery inspect
    inspect = celery_app.control.inspect()
    active_tasks = inspect.active()

    if not active_tasks:
        return []

    tasks = []
    for worker, task_list in active_tasks.items():
        for task in task_list:
            tasks.append(
                {
                    "task_id": task["id"],
                    "name": task["name"],
                    "worker": worker,
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {}),
                }
            )

    return tasks
