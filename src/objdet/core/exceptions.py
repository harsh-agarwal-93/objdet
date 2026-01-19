"""Custom exception hierarchy for ObjDet.

This module provides a structured exception hierarchy for explicit error handling
throughout the framework. Each exception type corresponds to a specific error domain,
making debugging and error recovery more straightforward.

Example:
    >>> from objdet.core.exceptions import ModelError, ConfigurationError
    >>>
    >>> try:
    ...     load_model("invalid_model")
    ... except ModelError as e:
    ...     logger.error(f"Failed to load model: {e}")
    ...     raise
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ObjDetError(Exception):
    """Base exception for all ObjDet errors.

    All custom exceptions in the framework inherit from this class,
    allowing for broad exception catching when needed.

    Args:
        message: Human-readable error description.
        details: Optional dict with additional context.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation with details if present."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigurationError(ObjDetError):
    """Error in configuration parsing or validation.

    Raised when:
    - YAML/config file syntax is invalid
    - Required configuration keys are missing
    - Configuration values fail validation
    - Conflicting configuration options are specified
    """

    pass


class DataError(ObjDetError):
    """Error related to data loading or processing.

    Raised when:
    - Dataset files are missing or corrupted
    - Data format is invalid or unsupported
    - Annotations are malformed
    - Transform operations fail
    """

    pass


class DataFormatError(DataError):
    """Error specific to dataset format issues.

    Args:
        message: Error description.
        format_name: Name of the dataset format (e.g., "coco", "voc").
        file_path: Path to the problematic file.
    """

    def __init__(
        self,
        message: str,
        format_name: str | None = None,
        file_path: Path | str | None = None,
    ) -> None:
        details = {}
        if format_name:
            details["format"] = format_name
        if file_path:
            details["file"] = str(file_path)
        super().__init__(message, details)
        self.format_name = format_name
        self.file_path = file_path


class ClassMappingError(DataError):
    """Error in class index mapping between formats.

    Raised when class indices cannot be properly mapped between
    YOLO format (no background) and TorchVision format (background at 0).
    """

    pass


class ModelError(ObjDetError):
    """Error related to model creation or loading.

    Raised when:
    - Model architecture is not found in registry
    - Model weights fail to load
    - Model parameters are invalid
    - Checkpoint is corrupted or incompatible
    """

    pass


class ModelNotFoundError(ModelError):
    """Requested model is not registered.

    Args:
        message: Error description.
        model_name: Name of the model that was not found.
        available_models: List of available model names.
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        details = {}
        if model_name:
            details["requested_model"] = model_name
        if available_models:
            details["available_models"] = ", ".join(available_models)
        super().__init__(message, details)
        self.model_name = model_name
        self.available_models = available_models


class CheckpointError(ModelError):
    """Error loading or saving model checkpoint.

    Args:
        message: Error description.
        checkpoint_path: Path to the checkpoint file.
    """

    def __init__(
        self,
        message: str,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        details = {}
        if checkpoint_path:
            details["checkpoint"] = str(checkpoint_path)
        super().__init__(message, details)
        self.checkpoint_path = checkpoint_path


class TrainingError(ObjDetError):
    """Error during model training.

    Raised when:
    - Training fails to start
    - Loss becomes NaN or Inf
    - Memory allocation fails
    - Distributed training synchronization fails
    """

    pass


class InferenceError(ObjDetError):
    """Error during model inference.

    Raised when:
    - Input preprocessing fails
    - Model forward pass fails
    - Output postprocessing fails
    - Batch processing fails
    """

    pass


class OptimizationError(ObjDetError):
    """Error during model optimization or export.

    Raised when:
    - ONNX export fails
    - TensorRT optimization fails
    - Model quantization fails
    - SafeTensors serialization fails
    """

    pass


class ExportError(OptimizationError):
    """Error exporting model to different format.

    Args:
        message: Error description.
        source_format: Original model format.
        target_format: Target export format (e.g., "onnx", "tensorrt").
    """

    def __init__(
        self,
        message: str,
        source_format: str | None = None,
        target_format: str | None = None,
    ) -> None:
        details = {}
        if source_format:
            details["source_format"] = source_format
        if target_format:
            details["target_format"] = target_format
        super().__init__(message, details)
        self.source_format = source_format
        self.target_format = target_format


class ServingError(ObjDetError):
    """Error in model serving/deployment.

    Raised when:
    - Server fails to start
    - Model loading for serving fails
    - Request processing fails
    - A/B testing configuration is invalid
    """

    pass


class PipelineError(ObjDetError):
    """Error in job pipeline execution.

    Raised when:
    - Job submission fails
    - Worker connection fails
    - Job dependency resolution fails
    - Job execution times out
    """

    pass


class JobSubmissionError(PipelineError):
    """Error submitting job to queue.

    Args:
        message: Error description.
        job_id: ID of the failed job.
        queue_name: Name of the target queue.
    """

    def __init__(
        self,
        message: str,
        job_id: str | None = None,
        queue_name: str | None = None,
    ) -> None:
        details = {}
        if job_id:
            details["job_id"] = job_id
        if queue_name:
            details["queue"] = queue_name
        super().__init__(message, details)
        self.job_id = job_id
        self.queue_name = queue_name


class ResourceRoutingError(PipelineError):
    """Error routing job to appropriate worker.

    Args:
        message: Error description.
        required_resources: Resources requested by the job.
        available_resources: Resources available on workers.
    """

    def __init__(
        self,
        message: str,
        required_resources: dict | None = None,
        available_resources: dict | None = None,
    ) -> None:
        details = {}
        if required_resources:
            details["required"] = str(required_resources)
        if available_resources:
            details["available"] = str(available_resources)
        super().__init__(message, details)
        self.required_resources = required_resources
        self.available_resources = available_resources
