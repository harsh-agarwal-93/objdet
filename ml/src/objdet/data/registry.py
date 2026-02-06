"""Data registries for ObjDet.

This module provides registries for data modules and transforms,
enabling plugin-style extensibility.
"""

from objdet.core.registry import Registry

# Registry for data modules (COCO, VOC, YOLO, etc.)
DATAMODULE_REGISTRY: Registry = Registry("datamodules")

# Registry for transforms
TRANSFORM_REGISTRY: Registry = Registry("transforms")

__all__ = ["DATAMODULE_REGISTRY", "TRANSFORM_REGISTRY"]
