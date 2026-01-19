"""Ensemble model implementations.

This package provides ensemble strategies for combining predictions
from multiple object detection models:
- Weighted Box Fusion (WBF)
- Non-Maximum Suppression (NMS) ensembling
- Soft-NMS ensembling
- Learned ensemble with trainable head
"""

from objdet.models.ensemble.base import BaseEnsemble
from objdet.models.ensemble.wbf import WBFEnsemble
from objdet.models.ensemble.nms import NMSEnsemble

__all__ = ["BaseEnsemble", "WBFEnsemble", "NMSEnsemble"]
