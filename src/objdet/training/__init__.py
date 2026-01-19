"""Training infrastructure: callbacks, metrics, losses.

This package provides training utilities for object detection:
- Custom callbacks for visualization and logging
- Detection-specific metrics
- Loss functions
"""

from objdet.training.callbacks import (
    ConfusionMatrixCallback,
    DetectionVisualizationCallback,
    GradientMonitorCallback,
    LearningRateMonitorCallback,
)
from objdet.training.metrics import (
    ConfusionMatrix,
    ClasswiseAP,
)

__all__ = [
    # Callbacks
    "ConfusionMatrixCallback",
    "DetectionVisualizationCallback",
    "GradientMonitorCallback",
    "LearningRateMonitorCallback",
    # Metrics
    "ConfusionMatrix",
    "ClasswiseAP",
]
