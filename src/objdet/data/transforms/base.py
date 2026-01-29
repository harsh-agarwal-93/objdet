"""Base transform classes for object detection.

These transforms handle both image and target (bounding boxes)
transformations together to ensure consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    from objdet.core.types import DetectionTarget

logger = get_logger(__name__)


class BaseTransform(ABC):
    """Abstract base class for detection transforms.

    Detection transforms must handle both the image and the target
    (bounding boxes and labels) together.
    """

    @abstractmethod
    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Apply transform to image and target.

        Args:
            image: Image tensor (C, H, W).
            target: Target dictionary with boxes, labels, etc.

        Returns:
            Tuple of (transformed_image, transformed_target).
        """
        ...


class Compose(BaseTransform):
    """Compose multiple transforms together.

    Args:
        transforms: List of transforms to apply in order.

    Example:
        >>> transforms = Compose(
        ...     [
        ...         Resize(800, 1333),
        ...         RandomHorizontalFlip(p=0.5),
        ...         Normalize(),
        ...     ]
        ... )
    """

    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        """String representation."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class ToTensor(BaseTransform):
    """Ensure image is a float tensor in [0, 1] range.

    If image is already a tensor, this is a no-op.
    If image is a numpy array, convert to tensor.
    """

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Convert to tensor if needed."""
        if not isinstance(image, Tensor):
            image = torch.from_numpy(image).float()

        # Ensure float in [0, 1]
        if image.dtype != torch.float32:
            image = image.float()

        if image.max() > 1.0:
            image = image / 255.0

        return image, target


class Normalize(BaseTransform):
    """Normalize image with ImageNet mean and std.

    Args:
        mean: Channel means (default: ImageNet).
        std: Channel standard deviations (default: ImageNet).
    """

    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Normalize image channels."""
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        image = (image - mean) / std
        return image, target


class Resize(BaseTransform):
    """Resize image and bounding boxes.

    Args:
        min_size: Minimum size of the shorter side.
        max_size: Maximum size of the longer side.
        mode: Interpolation mode.
    """

    def __init__(
        self,
        min_size: int = 800,
        max_size: int = 1333,
        mode: str = "bilinear",
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.mode = mode

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Resize image maintaining aspect ratio."""
        _, h, w = image.shape

        # Calculate scale
        min_dim = min(h, w)
        max_dim = max(h, w)

        scale = self.min_size / min_dim
        if max_dim * scale > self.max_size:
            scale = self.max_size / max_dim

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode=self.mode,
            align_corners=False if self.mode != "nearest" else None,
        ).squeeze(0)

        # Scale bounding boxes
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= scale  # x coordinates
            boxes[:, [1, 3]] *= scale  # y coordinates
            target["boxes"] = boxes

            # Update area if present
            if "area" in target:
                target["area"] = target["area"] * (scale**2)

        return image, target
