"""Format parsers for different dataset types.

This package provides parsers and datasets for various annotation formats:
- COCO: JSON annotations
- LitData: Optimized streaming format
- VOC: Pascal VOC XML annotations
- YOLO: Text file annotations

Each format has a corresponding Dataset class and DataModule.
"""

from objdet.data.formats.coco import COCODataModule, COCODataset
from objdet.data.formats.litdata import LitDataDataModule, LitDataDataset
from objdet.data.formats.voc import VOCDataModule, VOCDataset
from objdet.data.formats.yolo import YOLODataModule, YOLODataset

__all__ = [
    "COCODataset",
    "COCODataModule",
    "LitDataDataset",
    "LitDataDataModule",
    "VOCDataset",
    "VOCDataModule",
    "YOLODataset",
    "YOLODataModule",
]
