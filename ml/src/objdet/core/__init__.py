"""Core utilities for ObjDet.

This package provides core functionality used throughout the framework:
- Logging: Centralized logging with loguru and rich
- Exceptions: Custom exception hierarchy for better error handling
- Registry: Plugin architecture for extensibility
- Types: Common type definitions
- Constants: Shared constants and enums
"""

from objdet.core.exceptions import (
    ConfigurationError,
    DataError,
    InferenceError,
    ModelError,
    ObjDetError,
    OptimizationError,
    PipelineError,
    ServingError,
    TrainingError,
)
from objdet.core.logging import configure_logging, get_logger

__all__ = [  # noqa: RUF022
    # Logging
    "configure_logging",
    "get_logger",
    # Exceptions
    "ObjDetError",
    "ConfigurationError",
    "DataError",
    "ModelError",
    "TrainingError",
    "InferenceError",
    "OptimizationError",
    "ServingError",
    "PipelineError",
]
