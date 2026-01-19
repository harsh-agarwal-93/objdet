"""Custom metrics for object detection.

This package provides detection-specific metrics:
- Confusion matrix metric
- Class-wise AP computation
"""

from objdet.training.metrics.classwise_ap import ClasswiseAP
from objdet.training.metrics.confusion_matrix import ConfusionMatrix

__all__ = [
    "ClasswiseAP",
    "ConfusionMatrix",
]
