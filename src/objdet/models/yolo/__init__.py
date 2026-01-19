"""YOLO model implementations wrapped for Lightning.

This package provides Lightning wrappers for Ultralytics YOLO models:
- YOLOv8
- YOLOv11

These models use the YOLO class index mode (no background class).

IMPORTANT: Unlike the standard Ultralytics training flow that uses data.yaml,
these wrappers integrate with Lightning's DataModule system for full
ecosystem compatibility (callbacks, loggers, distributed training).
"""

from objdet.models.yolo.base import YOLOBaseLightning
from objdet.models.yolo.yolov8 import YOLOv8
from objdet.models.yolo.yolov11 import YOLOv11

__all__ = ["YOLOBaseLightning", "YOLOv8", "YOLOv11"]
