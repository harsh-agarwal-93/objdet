"""Unit tests for training API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient


def test_submit_training_job_success(
    test_client: TestClient,
    mock_celery_service: MagicMock,
    sample_training_config: dict[str, Any],
) -> None:
    """Test successful training job submission."""
    response = test_client.post("/api/training/submit", json=sample_training_config)

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id-123"
    assert data["status"] == "PENDING"
    assert "created_at" in data
    assert "estimated_duration" in data

    # Verify celery service was called
    mock_celery_service.submit_training_job.assert_called_once()


def test_submit_training_job_failure(
    test_client: TestClient,
    mock_celery_service: MagicMock,
    sample_training_config: dict[str, Any],
) -> None:
    """Test training job submission failure."""
    mock_celery_service.submit_training_job.side_effect = Exception("Celery connection error")

    response = test_client.post("/api/training/submit", json=sample_training_config)

    assert response.status_code == 500
    assert "Failed to submit training job" in response.json()["detail"]


def test_get_training_status(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test getting training task status."""
    response = test_client.get("/api/training/status/test-task-id-123")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id-123"
    assert data["status"] == "PENDING"

    mock_celery_service.get_task_status.assert_called_once_with("test-task-id-123")


def test_get_training_status_error(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test getting training status with error."""
    mock_celery_service.get_task_status.side_effect = Exception("Task not found")

    response = test_client.get("/api/training/status/nonexistent")

    assert response.status_code == 500
    assert "Failed to get task status" in response.json()["detail"]


def test_cancel_training_job(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test canceling a training job."""
    response = test_client.post("/api/training/cancel/test-task-id-123")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
    assert data["task_id"] == "test-task-id-123"

    mock_celery_service.cancel_task.assert_called_once_with("test-task-id-123")


def test_cancel_training_job_error(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test canceling a job with error."""
    mock_celery_service.cancel_task.side_effect = Exception("Cannot cancel task")

    response = test_client.post("/api/training/cancel/test-task-id-123")

    assert response.status_code == 500
    assert "Failed to cancel task" in response.json()["detail"]


def test_list_active_training_jobs(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test listing active training jobs."""
    mock_celery_service.list_active_tasks.return_value = [
        {
            "task_id": "task-1",
            "name": "train_model",
            "worker": "worker1",
            "args": [],
            "kwargs": {},
        },
        {
            "task_id": "task-2",
            "name": "export_model",
            "worker": "worker2",
            "args": [],
            "kwargs": {},
        },
    ]

    response = test_client.get("/api/training/active")

    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) == 2
    assert data["tasks"][0]["task_id"] == "task-1"

    mock_celery_service.list_active_tasks.assert_called_once()


def test_list_active_training_jobs_empty(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test listing active jobs when none exist."""
    mock_celery_service.list_active_tasks.return_value = []

    response = test_client.get("/api/training/active")

    assert response.status_code == 200
    data = response.json()
    assert data["tasks"] == []


def test_list_active_training_jobs_error(
    test_client: TestClient,
    mock_celery_service: MagicMock,
) -> None:
    """Test listing active jobs with error."""
    mock_celery_service.list_active_tasks.side_effect = Exception("Worker connection error")

    response = test_client.get("/api/training/active")

    assert response.status_code == 500
    assert "Failed to list active tasks" in response.json()["detail"]
