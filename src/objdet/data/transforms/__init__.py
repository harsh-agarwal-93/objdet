"""Data transforms for augmentation and preprocessing.

This package provides transform pipelines for object detection:
- Albumentations-based transforms
- Standard transforms (resize, normalize, etc.)
- Detection-specific augmentations (mosaic, mixup)
"""

from objdet.data.transforms.base import (
    BaseTransform,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from objdet.data.transforms.detection import (
    ColorJitter,
    DetectionTransform,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

__all__ = [
    # Base
    "BaseTransform",
    "Compose",
    "ToTensor",
    "Normalize",
    "Resize",
    # Detection
    "DetectionTransform",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ColorJitter",
    "RandomCrop",
]
