"""Constants and enums for ObjDet.

This module provides shared constants, enumerations, and configuration
values used throughout the framework.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Final


# =============================================================================
# Class Index Handling
# =============================================================================


class ClassIndexMode(str, Enum):
    """Mode for handling class indices between different model formats.

    CRITICAL: This explicitly handles the difference between:
    - TorchVision models: Background class at index 0, user classes start at 1
    - YOLO models: No background class, user classes start at 0

    Users MUST specify this in their data configuration to avoid subtle bugs.

    Example:
        >>> from objdet.core.constants import ClassIndexMode
        >>>
        >>> # For Faster R-CNN or RetinaNet
        >>> mode = ClassIndexMode.TORCHVISION
        >>>
        >>> # For YOLOv8 or YOLOv11
        >>> mode = ClassIndexMode.YOLO
    """

    TORCHVISION = "torchvision"
    """Background at index 0, user classes start at index 1."""

    YOLO = "yolo"
    """No background class, user classes start at index 0."""

    def has_background(self) -> bool:
        """Check if this mode includes a background class.

        Returns:
            True if background class is present at index 0.
        """
        return self == ClassIndexMode.TORCHVISION

    def user_class_offset(self) -> int:
        """Get the offset for user class indices.

        Returns:
            1 for TORCHVISION (background at 0), 0 for YOLO.
        """
        return 1 if self == ClassIndexMode.TORCHVISION else 0


# =============================================================================
# Dataset Formats
# =============================================================================


class DatasetFormat(str, Enum):
    """Supported dataset annotation formats.

    Each format has different file structures and annotation schemas.
    The parser for each format is in objdet.data.formats.
    """

    COCO = "coco"
    """COCO JSON format with annotations in single JSON file."""

    VOC = "voc"
    """Pascal VOC XML format with per-image annotation files."""

    YOLO = "yolo"
    """YOLO txt format with per-image label files."""

    CUSTOM = "custom"
    """Custom format requiring user-defined parser."""


# =============================================================================
# Model Types
# =============================================================================


class ModelType(str, Enum):
    """Types of object detection models supported."""

    FASTER_RCNN = "faster_rcnn"
    """Two-stage detector with region proposal network."""

    RETINANET = "retinanet"
    """One-stage detector with focal loss."""

    YOLOV8 = "yolov8"
    """Ultralytics YOLOv8."""

    YOLOV11 = "yolov11"
    """Ultralytics YOLOv11."""

    ENSEMBLE = "ensemble"
    """Ensemble of multiple models."""


class EnsembleStrategy(str, Enum):
    """Ensemble strategies for combining predictions."""

    WBF = "wbf"
    """Weighted Box Fusion - merges overlapping boxes with weighted average."""

    NMS = "nms"
    """Non-Maximum Suppression - keeps highest scoring non-overlapping boxes."""

    SOFT_NMS = "soft_nms"
    """Soft-NMS - reduces scores of overlapping boxes instead of removing."""

    LEARNED = "learned"
    """Learned ensemble with trainable combination head."""


# =============================================================================
# Export Formats
# =============================================================================


class ExportFormat(str, Enum):
    """Supported model export formats."""

    ONNX = "onnx"
    """ONNX format for cross-platform deployment."""

    TENSORRT = "tensorrt"
    """TensorRT optimized engine for NVIDIA GPUs."""

    TORCHSCRIPT = "torchscript"
    """TorchScript for PyTorch deployment."""

    SAFETENSORS = "safetensors"
    """SafeTensors for secure weight storage."""


# =============================================================================
# Serving
# =============================================================================


class ServingMode(str, Enum):
    """Serving modes for inference API."""

    SINGLE = "single"
    """Single model serving."""

    AB_TEST = "ab_test"
    """A/B testing with multiple model variants."""

    SHADOW = "shadow"
    """Shadow deployment for comparison testing."""


# =============================================================================
# Pipeline
# =============================================================================


class JobStatus(str, Enum):
    """Status of a pipeline job."""

    PENDING = "pending"
    """Job is queued and waiting for a worker."""

    RUNNING = "running"
    """Job is currently executing."""

    COMPLETED = "completed"
    """Job completed successfully."""

    FAILED = "failed"
    """Job failed with an error."""

    CANCELLED = "cancelled"
    """Job was cancelled by user."""


class JobPriority(int, Enum):
    """Priority levels for job queue."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


# =============================================================================
# Default Values
# =============================================================================

# Image processing
DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (640, 640)
DEFAULT_IMAGE_MEAN: Final[tuple[float, ...]] = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD: Final[tuple[float, ...]] = (0.229, 0.224, 0.225)

# Training
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4

# Inference
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.25
DEFAULT_NMS_THRESHOLD: Final[float] = 0.45
DEFAULT_MAX_DETECTIONS: Final[int] = 300

# Serving
DEFAULT_SERVE_PORT: Final[int] = 8000
DEFAULT_MAX_BATCH_SIZE: Final[int] = 32
DEFAULT_BATCH_TIMEOUT_MS: Final[int] = 10

# MLflow
DEFAULT_MLFLOW_EXPERIMENT: Final[str] = "objdet"
MLFLOW_STAGING_STAGE: Final[str] = "Staging"
MLFLOW_PRODUCTION_STAGE: Final[str] = "Production"

# File extensions
CHECKPOINT_EXTENSIONS: Final[tuple[str, ...]] = (".ckpt", ".pt", ".pth")
IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
