"""Unit tests for backend HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import polars as pl
import pytest
import respx

if TYPE_CHECKING:
    from frontend.api.client import BackendClient


@pytest.fixture
def backend_url() -> str:
    """Backend URL for testing.

    Returns:
        Test backend URL.
    """
    return "http://testserver:8000"


@pytest.fixture
def test_backend_client(backend_url: str) -> BackendClient:
    """Create test backend client.

    Args:
        backend_url: Test backend URL.

    Returns:
        Backend client instance.
    """
    from frontend.api.client import BackendClient

    return BackendClient(base_url=backend_url)


def test_client_initialization(backend_url: str, test_backend_client: BackendClient) -> None:
    """Test client initialization with custom URL."""
    assert test_backend_client.base_url == backend_url


@respx.mock
def test_submit_training_job(test_backend_client: BackendClient) -> None:
    """Test submitting a training job."""
    mock_response = {
        "task_id": "task-123",
        "status": "PENDING",
        "created_at": "2026-02-03T20:00:00",
        "estimated_duration": "20min",
    }

    respx.post("http://testserver:8000/api/training/submit").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    config = {"name": "test_job", "model_architecture": "yolov8", "dataset": "coco"}
    response = test_backend_client.submit_training_job(config)

    assert response["task_id"] == "task-123"
    assert response["status"] == "PENDING"


@respx.mock
def test_get_task_status(test_backend_client: BackendClient) -> None:
    """Test getting task status."""
    mock_response = {
        "task_id": "task-123",
        "status": "SUCCESS",
        "result": {"checkpoint": "/path/to/model.pt"},
        "error": None,
        "progress": None,
    }

    respx.get("http://testserver:8000/api/training/status/task-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.get_task_status("task-123")

    assert response["task_id"] == "task-123"
    assert response["status"] == "SUCCESS"
    assert response["result"]["checkpoint"] == "/path/to/model.pt"


@respx.mock
def test_cancel_task(test_backend_client: BackendClient) -> None:
    """Test canceling a task."""
    mock_response = {"status": "cancelled", "task_id": "task-123"}

    respx.post("http://testserver:8000/api/training/cancel/task-123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.cancel_task("task-123")

    assert response["status"] == "cancelled"
    assert response["task_id"] == "task-123"


@respx.mock
def test_list_active_tasks(test_backend_client: BackendClient) -> None:
    """Test listing active tasks."""
    mock_response = {
        "tasks": [
            {"task_id": "task-1", "name": "train_model"},
            {"task_id": "task-2", "name": "export_model"},
        ]
    }

    respx.get("http://testserver:8000/api/training/active").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.list_active_tasks()

    assert len(response) == 2
    assert response[0]["task_id"] == "task-1"


@respx.mock
def test_list_experiments(test_backend_client: BackendClient) -> None:
    """Test listing MLFlow experiments."""
    mock_response = {
        "experiments": [
            {
                "experiment_id": "1",
                "name": "objdet",
                "artifact_location": "/tmp/mlruns/1",
            }
        ]
    }

    respx.get("http://testserver:8000/api/mlflow/experiments").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.list_experiments()

    assert len(response) == 1
    assert response[0]["name"] == "objdet"


@respx.mock
def test_list_runs(test_backend_client: BackendClient) -> None:
    """Test listing runs with query parameters."""
    mock_response = {
        "runs": [
            {
                "run_id": "abc123",
                "run_name": "test_run",
                "status": "FINISHED",
            }
        ]
    }

    route = respx.get("http://testserver:8000/api/mlflow/runs").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.list_runs(experiment_id="1", status="FINISHED", max_results=50)

    assert len(response) == 1
    assert route.called
    # Verify query params were passed
    assert route.calls.last.request.url.params["experiment_id"] == "1"
    assert route.calls.last.request.url.params["status"] == "FINISHED"


@respx.mock
def test_get_run_details(test_backend_client: BackendClient) -> None:
    """Test getting run details."""
    mock_response = {
        "run_id": "abc123",
        "run_name": "test_run",
        "metrics": {"loss": 0.5},
        "params": {"epochs": "10"},
        "tags": {"mlflow.runName": "test_run"},
    }

    respx.get("http://testserver:8000/api/mlflow/runs/abc123").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.get_run_details("abc123")

    assert response["run_id"] == "abc123"
    assert response["metrics"]["loss"] == 0.5


@respx.mock
def test_get_run_metrics(test_backend_client: BackendClient) -> None:
    """Test getting run metrics as Polars DataFrame."""
    mock_response = {
        "metrics": [
            {"step": 0, "metric": "loss", "value": 1.0, "timestamp": 1234567890},
            {"step": 1, "metric": "loss", "value": 0.8, "timestamp": 1234567891},
        ]
    }

    respx.get("http://testserver:8000/api/mlflow/runs/abc123/metrics").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.get_run_metrics("abc123")

    assert isinstance(response, pl.DataFrame)
    assert len(response) == 2
    assert "metric" in response.columns
    assert "value" in response.columns


@respx.mock
def test_get_run_metrics_empty(test_backend_client: BackendClient) -> None:
    """Test getting metrics when empty."""
    mock_response = {"metrics": []}

    respx.get("http://testserver:8000/api/mlflow/runs/abc123/metrics").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.get_run_metrics("abc123")

    assert isinstance(response, pl.DataFrame)
    assert len(response) == 0


@respx.mock
def test_list_artifacts(test_backend_client: BackendClient) -> None:
    """Test listing artifacts."""
    mock_response = {"artifacts": [{"path": "model.pt", "is_dir": False, "file_size": 1024}]}

    route = respx.get("http://testserver:8000/api/mlflow/runs/abc123/artifacts").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.list_artifacts("abc123", path="models")

    assert len(response) == 1
    assert response[0]["path"] == "model.pt"
    # Verify path parameter was sent
    assert route.calls.last.request.url.params.get("path") == "models"


@respx.mock
def test_get_system_status(test_backend_client: BackendClient) -> None:
    """Test getting system status."""
    mock_response = {
        "celery_connected": True,
        "mlflow_connected": True,
        "active_workers": 2,
    }

    respx.get("http://testserver:8000/api/system/status").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = test_backend_client.get_system_status()

    assert response["celery_connected"] is True
    assert response["mlflow_connected"] is True


@respx.mock
def test_error_handling(test_backend_client: BackendClient) -> None:
    """Test HTTP error handling."""
    respx.get("http://testserver:8000/api/mlflow/runs/nonexistent").mock(
        return_value=httpx.Response(404, json={"detail": "Run not found"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        test_backend_client.get_run_details("nonexistent")


@respx.mock
def test_timeout_handling(backend_url: str) -> None:
    """Test timeout configuration."""
    from frontend.api.client import BackendClient

    client = BackendClient(base_url=backend_url)
    assert client.client.timeout.read == 30.0


def test_get_client_singleton() -> None:
    """Test singleton pattern for get_client."""
    from frontend.api.client import get_client

    client1 = get_client()
    client2 = get_client()

    assert client1 is client2
