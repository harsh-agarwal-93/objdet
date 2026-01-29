"""Data preprocessing utilities including LitData conversion.

This package provides utilities for preprocessing datasets:
- LitData format conversion for optimized streaming
- Dataset format conversion (COCO <-> YOLO <-> VOC)
"""

from objdet.data.preprocessing.litdata_converter import (
    LitDataConverter,
    convert_to_litdata,
)

__all__ = [
    "convert_to_litdata",
    "LitDataConverter",
]
