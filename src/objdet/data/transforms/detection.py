"""Detection-specific transforms with box handling.

These transforms are designed for object detection tasks and
properly transform bounding boxes along with the image.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from objdet.core.logging import get_logger
from objdet.data.transforms.base import BaseTransform

if TYPE_CHECKING:
    from objdet.core.types import DetectionTarget

logger = get_logger(__name__)


class DetectionTransform(BaseTransform):
    """Base class for detection-aware transforms.

    Provides utility methods for transforming bounding boxes.
    """

    @staticmethod
    def clip_boxes(
        boxes: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """Clip bounding boxes to image boundaries.

        Args:
            boxes: Boxes in xyxy format (N, 4).
            height: Image height.
            width: Image width.

        Returns:
            Clipped boxes.
        """
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        return boxes

    @staticmethod
    def remove_small_boxes(
        target: DetectionTarget,
        min_size: float = 1.0,
    ) -> DetectionTarget:
        """Remove boxes smaller than min_size.

        Args:
            target: Target dictionary.
            min_size: Minimum box size (width or height).

        Returns:
            Filtered target.
        """
        boxes = target["boxes"]
        if boxes.numel() == 0:
            return target

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= min_size) & (heights >= min_size)

        return {
            "boxes": boxes[keep],
            "labels": target["labels"][keep],
            "area": target.get("area", widths * heights)[keep],
            "iscrowd": target.get("iscrowd", torch.zeros(len(boxes), dtype=torch.int64))[keep],
            "image_id": target.get("image_id", 0),
        }


class RandomHorizontalFlip(DetectionTransform):
    """Randomly flip image and boxes horizontally.

    Args:
        p: Probability of applying the transform.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Apply random horizontal flip."""
        if random.random() < self.p:
            # Flip image
            image = image.flip(-1)  # Flip width dimension

            # Flip boxes
            _, _, w = image.shape
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

        return image, target


class RandomVerticalFlip(DetectionTransform):
    """Randomly flip image and boxes vertically.

    Args:
        p: Probability of applying the transform.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Apply random vertical flip."""
        if random.random() < self.p:
            # Flip image
            image = image.flip(-2)  # Flip height dimension

            # Flip boxes
            _, h, _ = image.shape
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes

        return image, target


class ColorJitter(DetectionTransform):
    """Randomly adjust brightness, contrast, saturation, and hue.

    Args:
        brightness: Brightness adjustment factor range [1-b, 1+b].
        contrast: Contrast adjustment factor range [1-c, 1+c].
        saturation: Saturation adjustment factor range [1-s, 1+s].
        hue: Hue adjustment range [-h, h].
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Apply color jitter (boxes unchanged)."""
        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = (image * factor).clamp(0, 1)

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = ((image - mean) * factor + mean).clamp(0, 1)

        # Note: Full saturation/hue adjustments require HSV conversion
        # This is a simplified version; for full implementation use torchvision

        return image, target


class RandomCrop(DetectionTransform):
    """Randomly crop image and adjust boxes.

    Args:
        min_scale: Minimum crop scale (fraction of original size).
        max_scale: Maximum crop scale.
        min_boxes_kept: Minimum fraction of boxes to keep (abort crop if not met).
    """

    def __init__(
        self,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        min_boxes_kept: float = 0.3,
    ) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_boxes_kept = min_boxes_kept

    def __call__(
        self,
        image: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        """Apply random crop."""
        _, h, w = image.shape

        # Random crop size
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        # Random crop position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # Check if enough boxes are kept
        boxes = target["boxes"]
        if boxes.numel() > 0:
            # Check box centers
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

            in_crop = (
                (centers_x >= left)
                & (centers_x < left + crop_w)
                & (centers_y >= top)
                & (centers_y < top + crop_h)
            )

            kept_ratio = in_crop.float().mean().item()
            if kept_ratio < self.min_boxes_kept:
                # Don't apply crop
                return image, target

            # Filter and adjust boxes
            new_boxes = boxes[in_crop].clone()
            new_boxes[:, [0, 2]] -= left
            new_boxes[:, [1, 3]] -= top
            new_boxes = self.clip_boxes(new_boxes, crop_h, crop_w)

            target = {
                "boxes": new_boxes,
                "labels": target["labels"][in_crop],
                "area": (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1]),
                "iscrowd": target.get("iscrowd", torch.zeros(len(boxes), dtype=torch.int64))[
                    in_crop
                ],
                "image_id": target.get("image_id", 0),
            }

        # Crop image
        image = image[:, top : top + crop_h, left : left + crop_w]

        return image, target


def get_train_transforms(
    min_size: int = 800,
    max_size: int = 1333,
    use_augmentation: bool = True,
) -> BaseTransform:
    """Get standard training transforms.

    Args:
        min_size: Minimum size for resize.
        max_size: Maximum size for resize.
        use_augmentation: Whether to apply augmentations.

    Returns:
        Composed transform pipeline.
    """
    from objdet.data.transforms.base import Compose, Normalize, Resize

    transforms: list[BaseTransform] = []

    if use_augmentation:
        transforms.extend(
            [
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    transforms.extend(
        [
            Resize(min_size=min_size, max_size=max_size),
            Normalize(),
        ]
    )

    return Compose(transforms)


def get_val_transforms(
    min_size: int = 800,
    max_size: int = 1333,
) -> BaseTransform:
    """Get standard validation/test transforms.

    Args:
        min_size: Minimum size for resize.
        max_size: Maximum size for resize.

    Returns:
        Composed transform pipeline.
    """
    from objdet.data.transforms.base import Compose, Normalize, Resize

    return Compose(
        [
            Resize(min_size=min_size, max_size=max_size),
            Normalize(),
        ]
    )
