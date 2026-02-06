"""Unit tests for Celery service."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_async_result() -> Mock:
    """Mock AsyncResult for testing.

    Returns:
        Mock AsyncResult instance.
    """
    return Mock()


@pytest.fixture
def mock_train_model_task() -> Mock:
    """Mock train_model task.

    Returns:
        Mock task instance.
    """
    mock_task = Mock()
    mock_task.delay.return_value.id = "task-123"
    return mock_task


def test_submit_training_job(mock_train_model_task: Mock) -> None:
    """Test submitting a training job."""
    from backend.services import celery_service

    with patch("backend.tasks.train_model", mock_train_model_task):
        task_id = celery_service.submit_training_job(
            config_path="/path/to/config.yaml",
            output_dir="/tmp/output",
            max_epochs=10,
            devices=1,
            accelerator="auto",
        )

        assert task_id == "task-123"
        mock_train_model_task.delay.assert_called_once_with(
            config_path="/path/to/config.yaml",
            output_dir="/tmp/output",
            max_epochs=10,
            devices=1,
            accelerator="auto",
        )


def test_get_task_status_pending() -> None:
    """Test getting status of a pending task."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.AsyncResult") as mock_async_result_class:
        mock_result = Mock()
        mock_result.state = "PENDING"
        mock_async_result_class.return_value = mock_result

        status = celery_service.get_task_status("task-123")

        assert status["task_id"] == "task-123"
        assert status["status"] == "PENDING"
        assert status["result"] is None
        assert status["error"] is None
        assert status["progress"] is None


def test_get_task_status_success() -> None:
    """Test getting status of a successful task."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.AsyncResult") as mock_async_result_class:
        mock_result = Mock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"checkpoint": "/path/to/best.pt", "final_loss": 0.123}
        mock_async_result_class.return_value = mock_result

        status = celery_service.get_task_status("task-123")

        assert status["task_id"] == "task-123"
        assert status["status"] == "SUCCESS"
        assert status["result"] == {"checkpoint": "/path/to/best.pt", "final_loss": 0.123}
        assert status["error"] is None


def test_get_task_status_failure() -> None:
    """Test getting status of a failed task."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.AsyncResult") as mock_async_result_class:
        mock_result = Mock()
        mock_result.state = "FAILURE"
        mock_result.info = Exception("Training failed: GPU out of memory")
        mock_async_result_class.return_value = mock_result

        status = celery_service.get_task_status("task-123")

        assert status["task_id"] == "task-123"
        assert status["status"] == "FAILURE"
        assert "GPU out of memory" in status["error"]


def test_get_task_status_progress() -> None:
    """Test getting status of a task in progress."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.AsyncResult") as mock_async_result_class:
        mock_result = Mock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"current": 5, "total": 10, "percent": 50}
        mock_async_result_class.return_value = mock_result

        status = celery_service.get_task_status("task-123")

        assert status["task_id"] == "task-123"
        assert status["status"] == "PROGRESS"
        assert status["progress"] == {"current": 5, "total": 10, "percent": 50}


def test_cancel_task() -> None:
    """Test canceling a running task."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.AsyncResult") as mock_async_result_class:
        mock_result = Mock()
        mock_async_result_class.return_value = mock_result

        result = celery_service.cancel_task("task-123")

        assert result is True
        mock_result.revoke.assert_called_once_with(terminate=True)


def test_list_active_tasks() -> None:
    """Test listing active tasks."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.celery_app") as mock_app:
        mock_inspect = Mock()
        mock_app.control.inspect.return_value = mock_inspect
        mock_inspect.active.return_value = {
            "worker1": [
                {
                    "id": "task-123",
                    "name": "objdet.pipelines.tasks.train_model",
                    "args": [],
                    "kwargs": {"config_path": "/path/to/config.yaml"},
                }
            ],
            "worker2": [
                {
                    "id": "task-456",
                    "name": "objdet.pipelines.tasks.export_model",
                    "args": [],
                    "kwargs": {"model_path": "/path/to/model.pt"},
                }
            ],
        }

        tasks = celery_service.list_active_tasks()

        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "task-123"
        assert tasks[0]["name"] == "objdet.pipelines.tasks.train_model"
        assert tasks[0]["worker"] == "worker1"
        assert tasks[1]["task_id"] == "task-456"
        assert tasks[1]["worker"] == "worker2"


def test_list_active_tasks_empty() -> None:
    """Test listing active tasks when none exist."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.celery_app") as mock_app:
        mock_inspect = Mock()
        mock_app.control.inspect.return_value = mock_inspect
        mock_inspect.active.return_value = None

        tasks = celery_service.list_active_tasks()

        assert tasks == []


def test_list_active_tasks_no_workers() -> None:
    """Test listing active tasks when workers have no tasks."""
    from backend.services import celery_service

    with patch("backend.services.celery_service.celery_app") as mock_app:
        mock_inspect = Mock()
        mock_app.control.inspect.return_value = mock_inspect
        mock_inspect.active.return_value = {}

        tasks = celery_service.list_active_tasks()

        assert tasks == []
