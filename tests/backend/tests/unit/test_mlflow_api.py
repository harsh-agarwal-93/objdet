"""Unit tests for MLFlow API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient


def test_list_experiments(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing MLFlow experiments."""
    response = test_client.get("/api/mlflow/experiments")

    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert len(data["experiments"]) == 1
    assert data["experiments"][0]["name"] == "objdet"

    mock_mlflow_service.list_experiments.assert_called_once()


def test_list_experiments_error(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing experiments with error."""
    mock_mlflow_service.list_experiments.side_effect = Exception("MLFlow connection error")

    response = test_client.get("/api/mlflow/experiments")

    assert response.status_code == 500
    assert "Failed to list experiments" in response.json()["detail"]


def test_list_runs_no_filter(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing runs without filters."""
    response = test_client.get("/api/mlflow/runs")

    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_id"] == "abc123"

    mock_mlflow_service.list_runs.assert_called_once_with(
        experiment_id=None,
        status=None,
        max_results=100,
    )


def test_list_runs_with_filters(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing runs with query parameters."""
    response = test_client.get(
        "/api/mlflow/runs",
        params={
            "experiment_id": "1",
            "status": "FINISHED",
            "max_results": 50,
        },
    )

    assert response.status_code == 200
    mock_mlflow_service.list_runs.assert_called_once_with(
        experiment_id="1",
        status="FINISHED",
        max_results=50,
    )


def test_list_runs_error(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing runs with error."""
    mock_mlflow_service.list_runs.side_effect = Exception("Query failed")

    response = test_client.get("/api/mlflow/runs")

    assert response.status_code == 500
    assert "Failed to list runs" in response.json()["detail"]


def test_get_run_details(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting run details."""
    response = test_client.get("/api/mlflow/runs/abc123")

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "abc123"
    assert data["run_name"] == "test_run"
    assert "metrics" in data
    assert "params" in data
    assert "tags" in data

    mock_mlflow_service.get_run_details.assert_called_once_with("abc123")


def test_get_run_details_error(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting run details with error."""
    mock_mlflow_service.get_run_details.side_effect = Exception("Run not found")

    response = test_client.get("/api/mlflow/runs/nonexistent")

    assert response.status_code == 500
    assert "Failed to get run details" in response.json()["detail"]


def test_get_run_metrics(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting run metrics."""
    response = test_client.get("/api/mlflow/runs/abc123/metrics")

    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert isinstance(data["metrics"], list)
    assert len(data["metrics"]) == 3  # From mock DataFrame

    mock_mlflow_service.get_run_metrics.assert_called_once_with("abc123")


def test_get_run_metrics_empty(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting run metrics when empty."""
    mock_mlflow_service.get_run_metrics.return_value = pl.DataFrame(
        {
            "step": [],
            "metric": [],
            "value": [],
            "timestamp": [],
        }
    )

    response = test_client.get("/api/mlflow/runs/abc123/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["metrics"] == []


def test_get_run_metrics_error(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting run metrics with error."""
    mock_mlflow_service.get_run_metrics.side_effect = Exception("Metrics not available")

    response = test_client.get("/api/mlflow/runs/abc123/metrics")

    assert response.status_code == 500
    assert "Failed to get run metrics" in response.json()["detail"]


def test_list_artifacts(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing artifacts."""
    response = test_client.get("/api/mlflow/runs/abc123/artifacts")

    assert response.status_code == 200
    data = response.json()
    assert "artifacts" in data
    assert len(data["artifacts"]) == 2
    assert data["artifacts"][0]["path"] == "model.pt"

    mock_mlflow_service.list_artifacts.assert_called_once_with("abc123", path="")


def test_list_artifacts_with_path(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing artifacts with path parameter."""
    response = test_client.get(
        "/api/mlflow/runs/abc123/artifacts",
        params={"path": "models/checkpoints"},
    )

    assert response.status_code == 200
    mock_mlflow_service.list_artifacts.assert_called_once_with(
        "abc123",
        path="models/checkpoints",
    )


def test_list_artifacts_error(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test listing artifacts with error."""
    mock_mlflow_service.list_artifacts.side_effect = Exception("Artifacts not found")

    response = test_client.get("/api/mlflow/runs/abc123/artifacts")

    assert response.status_code == 500
    assert "Failed to list artifacts" in response.json()["detail"]


def test_get_model_versions(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting model versions."""
    mock_mlflow_service.get_model_versions.return_value = [
        {
            "name": "yolov8",
            "version": "1",
            "creation_timestamp": 1234567890,
            "current_stage": "Production",
            "description": "YOLOv8 model",
        }
    ]

    response = test_client.get("/api/mlflow/models/yolov8/versions")

    assert response.status_code == 200
    data = response.json()
    assert "versions" in data
    assert len(data["versions"]) == 1
    assert data["versions"][0]["name"] == "yolov8"

    mock_mlflow_service.get_model_versions.assert_called_once_with("yolov8")


def test_get_model_versions_empty(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting model versions when none exist."""
    mock_mlflow_service.get_model_versions.return_value = []

    response = test_client.get("/api/mlflow/models/nonexistent/versions")

    assert response.status_code == 200
    data = response.json()
    assert data["versions"] == []


def test_get_model_versions_error(
    test_client: TestClient,
    mock_mlflow_service: MagicMock,
) -> None:
    """Test getting model versions with error."""
    mock_mlflow_service.get_model_versions.side_effect = Exception("Model registry error")

    response = test_client.get("/api/mlflow/models/yolov8/versions")

    assert response.status_code == 500
    assert "Failed to get model versions" in response.json()["detail"]
