"""Pytest configuration and fixtures for backend tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_celery_app() -> Mock:
    """Mock Celery application.

    Returns:
        Mock Celery app instance.
    """
    mock_app = Mock()
    mock_app.control.inspect.return_value.active.return_value = {}
    return mock_app


@pytest.fixture
def mock_mlflow_client() -> Mock:
    """Mock MLFlow client.

    Returns:
        Mock MLFlow client instance.
    """
    return Mock()


@pytest.fixture(autouse=True)
def mock_celery_inspect(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatically mock Celery inspection to prevent slow network calls.

    This fixture runs automatically for all tests to prevent real Celery
    broker connections during unit tests.

    Args:
        request: Pytest request fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    mock_inspect = Mock()
    mock_inspect.stats.return_value = {"worker1@host": {"uptime": 1000}}
    mock_inspect.active.return_value = {}

    mock_control = Mock()
    mock_control.inspect.return_value = mock_inspect

    # Mock the celery_app.control object to prevent real broker connections
    monkeypatch.setattr(
        "backend.services.celery_service.celery_app.control",
        mock_control,
        raising=False,
    )


@pytest.fixture(autouse=True)
def mock_mlflow_client_creation(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Automatically mock MLFlow client creation to prevent network calls.

    This fixture runs automatically for all tests to prevent real MLFlow
    connections during unit tests. It skips tests that explicitly test
    the client creation itself.

    Args:
        request: Pytest request fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    # Skip for tests that explicitly test the get_mlflow_client function
    if request.node.name == "test_get_mlflow_client":
        return

    mock_client = Mock()
    mock_client.search_experiments.return_value = []

    def mock_get_client(tracking_uri: str | None = None) -> Mock:
        """Mock get_mlflow_client function."""
        return mock_client

    monkeypatch.setattr(
        "backend.services.mlflow_service.get_mlflow_client",
        mock_get_client,
        raising=False,
    )


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """FastAPI test client.

    Yields:
        TestClient instance for making requests.
    """
    from backend.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_celery_service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock celery service module.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Mock celery service.
    """
    mock_service = MagicMock()

    # Default behavior for common methods
    mock_service.submit_training_job.return_value = "test-task-id-123"
    mock_service.get_task_status.return_value = {
        "task_id": "test-task-id-123",
        "status": "PENDING",
        "result": None,
        "error": None,
        "progress": None,
    }
    mock_service.cancel_task.return_value = True
    mock_service.list_active_tasks.return_value = []

    monkeypatch.setattr("backend.api.training.celery_service", mock_service)
    return mock_service


@pytest.fixture
def mock_mlflow_service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock MLFlow service module.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Mock MLFlow service.
    """
    import polars as pl

    mock_service = MagicMock()

    # Default behavior for common methods
    mock_service.list_experiments.return_value = [
        {
            "experiment_id": "1",
            "name": "objdet",
            "artifact_location": "/tmp/mlruns/1",
            "lifecycle_stage": "active",
            "tags": {},
        }
    ]

    mock_service.list_runs.return_value = [
        {
            "run_id": "abc123",
            "run_name": "test_run",
            "experiment_id": "1",
            "status": "FINISHED",
            "start_time": 1234567890,
            "end_time": 1234567900,
            "artifact_uri": "/tmp/mlruns/1/abc123/artifacts",
        }
    ]

    mock_service.get_run_details.return_value = {
        "run_id": "abc123",
        "run_name": "test_run",
        "experiment_id": "1",
        "status": "FINISHED",
        "start_time": 1234567890,
        "end_time": 1234567900,
        "artifact_uri": "/tmp/mlruns/1/abc123/artifacts",
        "metrics": {"loss": 0.5, "accuracy": 0.95},
        "params": {"epochs": "10", "batch_size": "32"},
        "tags": {"mlflow.runName": "test_run"},
    }

    mock_service.get_run_metrics.return_value = pl.DataFrame(
        {
            "step": [0, 1, 2],
            "metric": ["loss", "loss", "loss"],
            "value": [1.0, 0.8, 0.5],
            "timestamp": [1234567890, 1234567891, 1234567892],
        }
    )

    mock_service.list_artifacts.return_value = [
        {"path": "model.pt", "is_dir": False, "file_size": 1024},
        {"path": "config.yaml", "is_dir": False, "file_size": 256},
    ]

    mock_service.get_model_versions.return_value = []

    monkeypatch.setattr("backend.api.mlflow.mlflow_service", mock_service)
    return mock_service


@pytest.fixture
def sample_training_config() -> dict[str, Any]:
    """Sample training configuration for tests.

    Returns:
        Training configuration dictionary.
    """
    return {
        "name": "test_training_job",
        "model_architecture": "yolov8",
        "dataset": "coco",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "gpu": "auto",
        "priority": "normal",
        "mixed_precision": "fp16",
        "save_checkpoints": True,
        "early_stopping": True,
        "log_to_mlflow": True,
        "data_augmentation": True,
        "output_dir": None,
        "config_path": None,
    }


# Integration test fixtures (require live services)


@pytest.fixture(scope="session")
def check_services_available() -> bool:
    """Check if integration test services are available.

    Returns:
        True if services are available, False otherwise.
    """
    import os
    import socket

    # Check if we're in CI or explicitly skipping integration tests
    if os.getenv("SKIP_INTEGRATION_TESTS") == "1":
        return False

    # Check if RabbitMQ is accessible
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 5672))
        sock.close()
        if result != 0:
            return False
    except Exception:
        return False

    # Check if MLFlow is accessible
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 5000))
        sock.close()
        if result != 0:
            return False
    except Exception:
        return False

    return True


@pytest.fixture
def live_celery_app(check_services_available: bool) -> Any:
    """Get live Celery application for integration tests.

    Args:
        check_services_available: Fixture to check service availability.

    Returns:
        Live Celery app instance.

    Raises:
        pytest.skip: If services are not available.
    """
    if not check_services_available:
        pytest.skip("Integration test services not available")

    from backend.celery_app import app

    return app


@pytest.fixture
def live_mlflow_client(check_services_available: bool) -> Any:
    """Get live MLFlow client for integration tests.

    Args:
        check_services_available: Fixture to check service availability.

    Returns:
        Live MLFlow client instance.

    Raises:
        pytest.skip: If services are not available.
    """
    if not check_services_available:
        pytest.skip("Integration test services not available")

    from backend.core.config import settings
    from mlflow.client import MlflowClient

    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
