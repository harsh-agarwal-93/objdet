"""Inference utilities including SAHI integration.

This package provides:
- SAHI slicing for large image inference
- Batch prediction utilities
- Result post-processing
"""

from objdet.inference.predictor import Predictor
from objdet.inference.sahi_wrapper import SlicedInference

__all__ = [
    "Predictor",
    "SlicedInference",
]
