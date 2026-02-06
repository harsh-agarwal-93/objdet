"""Unit tests for MLFlow Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_experiment_model() -> None:
    """Test Experiment model creation and validation."""
    from backend.models.mlflow import Experiment

    # Valid experiment
    exp = Experiment(
        experiment_id="1",
        name="test_experiment",
        artifact_location="/tmp/mlruns/1",
        lifecycle_stage="active",
        tags={"env": "test"},
    )

    assert exp.experiment_id == "1"
    assert exp.name == "test_experiment"
    assert exp.lifecycle_stage == "active"
    assert exp.tags is not None
    assert exp.tags["env"] == "test"


def test_experiment_model_minimal() -> None:
    """Test Experiment model with minimal required fields."""
    from backend.models.mlflow import Experiment

    # Only required fields
    exp = Experiment(experiment_id="1", name="test")

    assert exp.experiment_id == "1"
    assert exp.name == "test"
    assert exp.artifact_location is None
    assert exp.lifecycle_stage is None
    assert exp.tags is None


def test_run_model() -> None:
    """Test Run model creation and validation."""
    from backend.models.mlflow import Run

    run = Run(
        run_id="abc123",
        run_name="test_run",
        experiment_id="1",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567900,
        artifact_uri="/tmp/artifacts",
    )

    assert run.run_id == "abc123"
    assert run.run_name == "test_run"
    assert run.status == "FINISHED"
    assert run.start_time == 1234567890


def test_run_model_optional_fields() -> None:
    """Test Run model with optional fields."""
    from backend.models.mlflow import Run

    # Only required fields
    run = Run(run_id="abc123", experiment_id="1", status="RUNNING")

    assert run.run_id == "abc123"
    assert run.run_name is None
    assert run.start_time is None
    assert run.end_time is None


def test_run_metrics_model() -> None:
    """Test RunMetrics model creation."""
    from backend.models.mlflow import RunMetrics

    metrics = RunMetrics(
        run_id="abc123",
        metrics=[
            {"step": 0, "metric": "loss", "value": 1.0, "timestamp": 1000},
            {"step": 1, "metric": "loss", "value": 0.8, "timestamp": 1001},
        ],
    )

    assert metrics.run_id == "abc123"
    assert len(metrics.metrics) == 2
    assert metrics.metrics[0]["metric"] == "loss"


def test_run_metrics_model_empty() -> None:
    """Test RunMetrics with empty metrics list."""
    from backend.models.mlflow import RunMetrics

    metrics = RunMetrics(run_id="abc123")

    assert metrics.run_id == "abc123"
    assert metrics.metrics == []  # Default empty list


def test_artifact_model() -> None:
    """Test Artifact model creation."""
    from backend.models.mlflow import Artifact

    artifact = Artifact(path="model.pt", is_dir=False, file_size=1024)

    assert artifact.path == "model.pt"
    assert artifact.is_dir is False
    assert artifact.file_size == 1024


def test_artifact_model_directory() -> None:
    """Test Artifact model for directory."""
    from backend.models.mlflow import Artifact

    artifact = Artifact(path="checkpoints/", is_dir=True)

    assert artifact.path == "checkpoints/"
    assert artifact.is_dir is True
    assert artifact.file_size is None  # Optional for directories


def test_model_version() -> None:
    """Test ModelVersion model creation."""
    from backend.models.mlflow import ModelVersion

    version = ModelVersion(
        name="yolov8",
        version="1",
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567900,
        current_stage="Production",
        description="YOLOv8 model",
        source="/tmp/models",
        run_id="abc123",
    )

    assert version.name == "yolov8"
    assert version.version == "1"
    assert version.current_stage == "Production"
    assert version.description == "YOLOv8 model"


def test_model_version_minimal() -> None:
    """Test ModelVersion with minimal required fields."""
    from backend.models.mlflow import ModelVersion

    version = ModelVersion(
        name="yolov8",
        version="1",
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567900,
        current_stage="None",
    )

    assert version.name == "yolov8"
    assert version.version == "1"
    assert version.description is None
    assert version.source is None
    assert version.run_id is None


def test_model_validation_error() -> None:
    """Test model validation errors with invalid data."""
    from backend.models.mlflow import Run

    # Missing required field
    with pytest.raises(ValidationError) as exc_info:
        Run(run_id="abc123")  # type: ignore[call-arg]

    error = exc_info.value
    assert len(error.errors()) >= 2  # At least 2 missing fields


def test_model_serialization() -> None:
    """Test model serialization to dict."""
    from backend.models.mlflow import Experiment

    exp = Experiment(experiment_id="1", name="test", tags={"key": "value"})

    data = exp.model_dump()

    assert isinstance(data, dict)
    assert data["experiment_id"] == "1"
    assert data["name"] == "test"
    assert data["tags"]["key"] == "value"


def test_model_deserialization() -> None:
    """Test model deserialization from dict."""
    from backend.models.mlflow import Run

    data = {"run_id": "abc", "experiment_id": "1", "status": "RUNNING"}

    run = Run(**data)

    assert run.run_id == "abc"
    assert run.experiment_id == "1"
    assert run.status == "RUNNING"
