"""TorchVision model implementations.

This package provides Lightning wrappers for TorchVision detection models:
- Faster R-CNN
- RetinaNet

These models use the TORCHVISION class index mode (background at index 0).
"""

from objdet.models.torchvision.faster_rcnn import FasterRCNN
from objdet.models.torchvision.retinanet import RetinaNet

__all__ = ["FasterRCNN", "RetinaNet"]
