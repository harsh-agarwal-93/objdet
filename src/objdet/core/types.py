"""Type definitions for ObjDet.

This module provides common type aliases and protocols used throughout
the framework for improved type safety and code clarity.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict, TypeVar, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image

# =============================================================================
# Basic Type Aliases
# =============================================================================

# Tensor types
BatchTensor: TypeAlias = Tensor  # (N, C, H, W) image batch
ImageTensor: TypeAlias = Tensor  # (C, H, W) single image
BoxTensor: TypeAlias = Tensor  # (N, 4) bounding boxes
LabelTensor: TypeAlias = Tensor  # (N,) class labels
ScoreTensor: TypeAlias = Tensor  # (N,) confidence scores
MaskTensor: TypeAlias = Tensor  # (N, H, W) segmentation masks

# Numpy types
if TYPE_CHECKING:
    NumpyImage: TypeAlias = NDArray[np.uint8]  # (H, W, C) image array
    NumpyBoxes: TypeAlias = NDArray[np.float32]  # (N, 4) bounding boxes

# Path types
PathLike: TypeAlias = str | Path

# Config types
ConfigDict: TypeAlias = dict[str, Any]
HyperParams: TypeAlias = Mapping[str, int | float | str | bool]

# =============================================================================
# Detection Types
# =============================================================================


class _DetectionTargetRequired(TypedDict):
    """Required fields for DetectionTarget."""

    boxes: Tensor
    labels: Tensor


class DetectionTarget(_DetectionTargetRequired, total=False):
    """Type definition for a detection annotation target.

    This follows the torchvision detection format.

    Attributes:
        boxes: (N, 4) tensor of bounding boxes in [x1, y1, x2, y2] format (required).
        labels: (N,) tensor of class labels (required).
        masks: Optional (N, H, W) tensor of instance masks.
        area: Optional (N,) tensor of box areas.
        iscrowd: Optional (N,) tensor indicating crowd regions.
        image_id: Optional image identifier.
    """

    masks: Tensor
    area: Tensor
    iscrowd: Tensor
    image_id: int


class DetectionPrediction(TypedDict):
    """Type definition for model detection output.

    Attributes:
        boxes: (N, 4) tensor of predicted bounding boxes.
        labels: (N,) tensor of predicted class labels.
        scores: (N,) tensor of confidence scores.
    """

    boxes: Tensor
    labels: Tensor
    scores: Tensor


# Batch types
DetectionBatch: TypeAlias = tuple[list[Tensor], list[DetectionTarget]]
PredictionBatch: TypeAlias = list[DetectionPrediction]

# =============================================================================
# Callback Types
# =============================================================================

MetricValue: TypeAlias = float | Tensor
MetricsDict: TypeAlias = dict[str, MetricValue]

# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class SupportsPredict(Protocol):
    """Protocol for objects that support prediction."""

    def predict(self, images: list[Tensor]) -> PredictionBatch:
        """Run prediction on images.

        Args:
            images: List of image tensors.

        Returns:
            List of prediction dictionaries.
        """
        ...


@runtime_checkable
class SupportsExport(Protocol):
    """Protocol for objects that support model export."""

    def export(self, path: PathLike, format: str) -> None:
        """Export model to specified format.

        Args:
            path: Output path for exported model.
            format: Export format (e.g., "onnx", "tensorrt").
        """
        ...


@runtime_checkable
class SupportsTransform(Protocol):
    """Protocol for data transform objects."""

    def __call__(
        self,
        image: Tensor | Image.Image,
        target: DetectionTarget | None = None,
    ) -> tuple[Tensor, DetectionTarget | None]:
        """Apply transform to image and target.

        Args:
            image: Input image.
            target: Optional detection target.

        Returns:
            Tuple of transformed image and target.
        """
        ...


# =============================================================================
# Generic Types
# =============================================================================

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="SupportsPredict")
TransformT = TypeVar("TransformT", bound="SupportsTransform")
