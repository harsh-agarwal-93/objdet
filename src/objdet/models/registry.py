"""Model registry for ObjDet.

This module provides the central registry for all detection models.
Models are registered using decorators and can be retrieved by name.

Example:
    >>> from objdet.models.registry import MODEL_REGISTRY
    >>>
    >>> # Register a new model
    >>> @MODEL_REGISTRY.register("my_detector")
    ... class MyDetector(BaseLightningDetector):
    ...     pass
    >>>
    >>> # Build a model from registry
    >>> model = MODEL_REGISTRY.build("faster_rcnn", num_classes=80)
"""

from objdet.core.registry import Registry
from objdet.models.base import BaseLightningDetector

# Global model registry
MODEL_REGISTRY: Registry[BaseLightningDetector] = Registry("models")

__all__ = ["MODEL_REGISTRY"]
