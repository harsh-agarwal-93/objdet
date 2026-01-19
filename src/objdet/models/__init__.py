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

# Import subpackages to register models
from objdet.models import torchvision as _tv  # noqa: F401
from objdet.models import yolo as _yolo  # noqa: F401
from objdet.models import ensemble as _ensemble  # noqa: F401

# Re-export commonly used models
from objdet.models.torchvision import FasterRCNN, RetinaNet
from objdet.models.yolo import YOLOv8, YOLOv11
from objdet.models.ensemble import WBFEnsemble, NMSEnsemble

__all__ = [
    # Base
    "BaseLightningDetector",
    "MODEL_REGISTRY",
    # TorchVision
    "FasterRCNN",
    "RetinaNet",
    # YOLO
    "YOLOv8",
    "YOLOv11",
    # Ensemble
    "WBFEnsemble",
    "NMSEnsemble",
]
