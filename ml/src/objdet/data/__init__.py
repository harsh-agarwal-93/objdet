"""Data package for ObjDet.

This package provides data loading infrastructure:
- BaseDataModule: Abstract Lightning DataModule for detection
- Format parsers: COCO, VOC, YOLO
- Transforms: Detection-aware augmentations
- Preprocessing: LitData conversion

The data package handles the critical class index mapping between
YOLO format (no background) and TorchVision format (background at 0).
"""

# Import format modules to register datamodules
from objdet.data import formats as _formats  # noqa: F401
from objdet.data.base import BaseDataModule, detection_collate_fn
from objdet.data.class_mapping import ClassMapper
from objdet.data.formats import (
    COCODataModule,
    COCODataset,
    VOCDataModule,
    VOCDataset,
    YOLODataModule,
    YOLODataset,
)

# Import preprocessing
from objdet.data.preprocessing import convert_to_litdata
from objdet.data.registry import DATAMODULE_REGISTRY, TRANSFORM_REGISTRY

# Import transforms
from objdet.data.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)
from objdet.data.transforms.detection import get_train_transforms, get_val_transforms

__all__ = [
    # Base
    "BaseDataModule",
    "detection_collate_fn",
    "ClassMapper",
    "DATAMODULE_REGISTRY",
    "TRANSFORM_REGISTRY",
    # Datasets
    "COCODataset",
    "COCODataModule",
    "VOCDataset",
    "VOCDataModule",
    "YOLODataset",
    "YOLODataModule",
    # Transforms
    "Compose",
    "Normalize",
    "Resize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ColorJitter",
    "get_train_transforms",
    "get_val_transforms",
    # Preprocessing
    "convert_to_litdata",
]
