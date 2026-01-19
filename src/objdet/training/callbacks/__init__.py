"""Custom callbacks for object detection training.

This package provides Lightning callbacks for:
- Confusion matrix generation and saving
- Detection visualization on sample images
- Gradient monitoring
- Learning rate logging
"""

from objdet.training.callbacks.confusion_matrix import ConfusionMatrixCallback
from objdet.training.callbacks.visualization import DetectionVisualizationCallback
from objdet.training.callbacks.gradient_monitor import GradientMonitorCallback
from objdet.training.callbacks.lr_monitor import LearningRateMonitorCallback

__all__ = [
    "ConfusionMatrixCallback",
    "DetectionVisualizationCallback",
    "GradientMonitorCallback",
    "LearningRateMonitorCallback",
]
