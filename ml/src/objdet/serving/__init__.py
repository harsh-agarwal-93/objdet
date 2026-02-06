"""LitServe-based model serving.

This package provides REST API deployment using Lightning LitServe:
- Detection API server
- A/B testing support
- Dynamic batching
"""

from objdet.serving.api import DetectionAPI
from objdet.serving.server import run_server

__all__ = [
    "DetectionAPI",
    "run_server",
]
