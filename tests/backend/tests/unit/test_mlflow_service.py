"""Unit tests for MLFlow service."""

from __future__ import annotations

from unittest.mock import Mock, patch

import polars as pl
import pytest


@pytest.fixture
def mock_mlflow_client_instance() -> Mock:
    """Mock MLFlow client instance with configured behavior.

    Returns:
        Mock MLFlow client.
    """
    mock_client = Mock()

    # Mock experiment
    mock_exp = Mock()
    mock_exp.experiment_id = "1"
    mock_exp.name = "objdet"
    mock_exp.artifact_location = "/tmp/mlruns/1"
    mock_exp.lifecycle_stage = "active"
    mock_exp.tags = {}

    mock_client.search_experiments.return_value = [mock_exp]
    mock_client.get_experiment_by_name.return_value = mock_exp

    # Mock run
    mock_run = Mock()
    mock_run.info.run_id = "abc123"
    mock_run.info.experiment_id = "1"
    mock_run.info.status = "FINISHED"
    mock_run.info.start_time = 1234567890
    mock_run.info.end_time = 1234567900
    mock_run.info.artifact_uri = "/tmp/mlruns/1/abc123/artifacts"
    mock_run.data.tags = {"mlflow.runName": "test_run"}
    mock_run.data.metrics = {"loss": 0.5, "accuracy": 0.95}
    mock_run.data.params = {"epochs": "10", "batch_size": "32"}

    mock_client.search_runs.return_value = [mock_run]
    mock_client.get_run.return_value = mock_run

    # Mock metrics history
    mock_metric = Mock()
    mock_metric.step = 0
    mock_metric.value = 1.0
    mock_metric.timestamp = 1234567890
    mock_client.get_metric_history.return_value = [mock_metric]

    # Mock artifacts
    mock_artifact = Mock()
    mock_artifact.path = "model.pt"
    mock_artifact.is_dir = False
    mock_artifact.file_size = 1024
    mock_client.list_artifacts.return_value = [mock_artifact]

    # Mock model versions
    mock_client.search_model_versions.return_value = []

    return mock_client


def test_get_mlflow_client() -> None:
    """Test getting MLFlow client."""
    from backend.services import mlflow_service

    with patch("backend.services.mlflow_service.MlflowClient") as mock_mlflow_client_class:
        client = mlflow_service.get_mlflow_client()

        mock_mlflow_client_class.assert_called_once()
        assert client is not None


def test_list_experiments(mock_mlflow_client_instance: Mock) -> None:
    """Test listing experiments."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        experiments = mlflow_service.list_experiments()

        assert len(experiments) == 1
        assert experiments[0]["experiment_id"] == "1"
        assert experiments[0]["name"] == "objdet"
        assert experiments[0]["artifact_location"] == "/tmp/mlruns/1"
        assert experiments[0]["lifecycle_stage"] == "active"


def test_list_runs_no_filter(mock_mlflow_client_instance: Mock) -> None:
    """Test listing runs without filters."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        runs = mlflow_service.list_runs()

        assert len(runs) == 1
        assert runs[0]["run_id"] == "abc123"
        assert runs[0]["run_name"] == "test_run"
        assert runs[0]["status"] == "FINISHED"


def test_list_runs_with_status_filter(mock_mlflow_client_instance: Mock) -> None:
    """Test listing runs with status filter."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        mlflow_service.list_runs(status="FINISHED")

        # Verify the filter string was constructed
        mock_mlflow_client_instance.search_runs.assert_called()
        call_kwargs = mock_mlflow_client_instance.search_runs.call_args.kwargs
        assert call_kwargs["filter_string"] == "attributes.status = 'FINISHED'"


def test_list_runs_with_experiment_id(mock_mlflow_client_instance: Mock) -> None:
    """Test listing runs with experiment ID filter."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        mlflow_service.list_runs(experiment_id="1")

        # Verify experiment_ids was passed
        mock_mlflow_client_instance.search_runs.assert_called()
        call_kwargs = mock_mlflow_client_instance.search_runs.call_args.kwargs
        assert call_kwargs["experiment_ids"] == ["1"]


def test_list_runs_no_experiment(mock_mlflow_client_instance: Mock) -> None:
    """Test listing runs when default experiment doesn't exist."""
    from backend.services import mlflow_service

    mock_mlflow_client_instance.get_experiment_by_name.return_value = None

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        runs = mlflow_service.list_runs()

        assert runs == []


def test_get_run_details(mock_mlflow_client_instance: Mock) -> None:
    """Test getting run details."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        details = mlflow_service.get_run_details("abc123")

        assert details["run_id"] == "abc123"
        assert details["run_name"] == "test_run"
        assert details["status"] == "FINISHED"
        assert details["metrics"] == {"loss": 0.5, "accuracy": 0.95}
        assert details["params"] == {"epochs": "10", "batch_size": "32"}
        assert "mlflow.runName" in details["tags"]


def test_get_run_metrics(mock_mlflow_client_instance: Mock) -> None:
    """Test getting run metrics as DataFrame."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        df = mlflow_service.get_run_metrics("abc123")

        assert isinstance(df, pl.DataFrame)
        assert "step" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns
        assert "timestamp" in df.columns
        assert len(df) == 2  # Two metrics (loss and accuracy) returned from mock


def test_get_run_metrics_empty(mock_mlflow_client_instance: Mock) -> None:
    """Test getting run metrics when no metrics exist."""
    from backend.services import mlflow_service

    # Mock a run with no metrics
    mock_run = Mock()
    mock_run.data.metrics = {}
    mock_mlflow_client_instance.get_run.return_value = mock_run

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        df = mlflow_service.get_run_metrics("abc123")

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "step" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns


def test_list_artifacts(mock_mlflow_client_instance: Mock) -> None:
    """Test listing artifacts for a run."""
    from backend.services import mlflow_service

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        artifacts = mlflow_service.list_artifacts("abc123", path="models")

        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "model.pt"
        assert artifacts[0]["is_dir"] is False
        assert artifacts[0]["file_size"] == 1024

        mock_mlflow_client_instance.list_artifacts.assert_called_once_with("abc123", path="models")


def test_get_model_versions(mock_mlflow_client_instance: Mock) -> None:
    """Test getting model versions."""
    from backend.services import mlflow_service

    # Mock model version
    mock_version = Mock()
    mock_version.name = "yolov8"
    mock_version.version = "1"
    mock_version.creation_timestamp = 1234567890
    mock_version.last_updated_timestamp = 1234567900
    mock_version.current_stage = "Production"
    mock_version.description = "YOLOv8 model"
    mock_version.source = "/tmp/mlruns/1/abc123/artifacts/model"
    mock_version.run_id = "abc123"

    mock_mlflow_client_instance.search_model_versions.return_value = [mock_version]

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        versions = mlflow_service.get_model_versions("yolov8")

        assert len(versions) == 1
        assert versions[0]["name"] == "yolov8"
        assert versions[0]["version"] == "1"
        assert versions[0]["current_stage"] == "Production"


def test_get_model_versions_not_found(mock_mlflow_client_instance: Mock) -> None:
    """Test getting model versions when model doesn't exist."""
    from backend.services import mlflow_service

    mock_mlflow_client_instance.search_model_versions.side_effect = Exception("Model not found")

    with patch(
        "backend.services.mlflow_service.get_mlflow_client",
        return_value=mock_mlflow_client_instance,
    ):
        versions = mlflow_service.get_model_versions("nonexistent")

        assert versions == []
