"""Models package for ObjDet.

This package provides object detection model implementations as
PyTorch Lightning modules, along with the model registry for
extensibility.

Available models:
- TorchVision: Faster R-CNN, RetinaNet
- YOLO: YOLOv8, YOLOv11 (Ultralytics wrappers)
- Ensemble: WBF, NMS, Learned combinations
"""

from objdet.models.base import BaseLightningDetector
from objdet.models.registry import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "BaseLightningDetector",
]
