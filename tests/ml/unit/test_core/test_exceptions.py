"""Unit tests for core exceptions module."""

from __future__ import annotations

from pathlib import Path

import pytest

from objdet.core.exceptions import (
    CheckpointError,
    ClassMappingError,
    ConfigurationError,
    DataError,
    DataFormatError,
    ExportError,
    InferenceError,
    JobSubmissionError,
    ModelError,
    ModelNotFoundError,
    ObjDetError,
    OptimizationError,
    PipelineError,
    ResourceRoutingError,
    ServingError,
    TrainingError,
)


class TestObjDetError:
    """Tests for base ObjDetError class."""

    def test_basic_message(self) -> None:
        """Test error with just a message."""
        error = ObjDetError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_message_with_details(self) -> None:
        """Test error with message and details."""
        error = ObjDetError("Failed to load", details={"path": "/foo/bar", "reason": "not found"})
        assert "Failed to load" in str(error)
        assert "path=/foo/bar" in str(error)
        assert "reason=not found" in str(error)
        assert error.details["path"] == "/foo/bar"

    def test_exception_inheritance(self) -> None:
        """Test that ObjDetError inherits from Exception."""
        error = ObjDetError("test")
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inheritance(self) -> None:
        """Test that ConfigurationError inherits from ObjDetError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, ObjDetError)

    def test_with_details(self) -> None:
        """Test ConfigurationError with details."""
        error = ConfigurationError(
            "Missing required key",
            details={"key": "model.num_classes"},
        )
        assert "model.num_classes" in str(error)


class TestDataFormatError:
    """Tests for DataFormatError."""

    def test_with_format_and_path(self) -> None:
        """Test DataFormatError with format and path."""
        error = DataFormatError(
            "Invalid annotation",
            format_name="coco",
            file_path=Path("/data/annotations.json"),
        )
        assert error.format_name == "coco"
        assert error.file_path == Path("/data/annotations.json")
        assert "coco" in str(error)
        assert "/data/annotations.json" in str(error)

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        error = DataFormatError("test")
        assert isinstance(error, DataError)
        assert isinstance(error, ObjDetError)


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_with_model_info(self) -> None:
        """Test ModelNotFoundError with model information."""
        error = ModelNotFoundError(
            "Model not found",
            model_name="yolov9",
            available_models=["faster_rcnn", "yolov8", "yolov11"],
        )
        assert error.model_name == "yolov9"
        assert error.available_models == ["faster_rcnn", "yolov8", "yolov11"]
        assert "yolov9" in str(error)

    def test_inheritance(self) -> None:
        """Test inheritance chain."""
        error = ModelNotFoundError("test")
        assert isinstance(error, ModelError)
        assert isinstance(error, ObjDetError)


class TestCheckpointError:
    """Tests for CheckpointError."""

    def test_with_path(self) -> None:
        """Test CheckpointError with checkpoint path."""
        error = CheckpointError(
            "Corrupted checkpoint",
            checkpoint_path="/models/best.ckpt",
        )
        assert error.checkpoint_path == "/models/best.ckpt"
        assert "/models/best.ckpt" in str(error)


class TestExportError:
    """Tests for ExportError."""

    def test_with_formats(self) -> None:
        """Test ExportError with source and target formats."""
        error = ExportError(
            "Export failed",
            source_format="pytorch",
            target_format="tensorrt",
        )
        assert error.source_format == "pytorch"
        assert error.target_format == "tensorrt"


class TestJobSubmissionError:
    """Tests for JobSubmissionError."""

    def test_with_job_info(self) -> None:
        """Test JobSubmissionError with job information."""
        error = JobSubmissionError(
            "Failed to submit",
            job_id="job-123",
            queue_name="gpu-queue",
        )
        assert error.job_id == "job-123"
        assert error.queue_name == "gpu-queue"


class TestResourceRoutingError:
    """Tests for ResourceRoutingError."""

    def test_with_resources(self) -> None:
        """Test ResourceRoutingError with resource information."""
        error = ResourceRoutingError(
            "No suitable worker",
            required_resources={"gpu": "A100", "count": 4},
            available_resources={"gpu": "T4", "count": 1},
        )
        assert error.required_resources == {"gpu": "A100", "count": 4}
        assert error.available_resources == {"gpu": "T4", "count": 1}


class TestExceptionHierarchy:
    """Tests for the exception hierarchy structure."""

    @pytest.mark.parametrize(
        "exception_class,parent_class",
        [
            (ConfigurationError, ObjDetError),
            (DataError, ObjDetError),
            (DataFormatError, DataError),
            (ClassMappingError, DataError),
            (ModelError, ObjDetError),
            (ModelNotFoundError, ModelError),
            (CheckpointError, ModelError),
            (TrainingError, ObjDetError),
            (InferenceError, ObjDetError),
            (OptimizationError, ObjDetError),
            (ExportError, OptimizationError),
            (ServingError, ObjDetError),
            (PipelineError, ObjDetError),
            (JobSubmissionError, PipelineError),
            (ResourceRoutingError, PipelineError),
        ],
    )
    def test_inheritance(
        self,
        exception_class: type[ObjDetError],
        parent_class: type[Exception],
    ) -> None:
        """Test that exception classes have correct inheritance."""
        assert issubclass(exception_class, parent_class)
        instance = exception_class("test message")
        assert isinstance(instance, parent_class)
