"""LitServe-based model serving.

This package provides REST API deployment using Lightning LitServe:
- Detection API server
- A/B testing support
- Dynamic batching
"""

from objdet.serving.server import run_server
from objdet.serving.api import DetectionAPI

__all__ = [
    "run_server",
    "DetectionAPI",
]
