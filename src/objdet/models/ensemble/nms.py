"""NMS-based ensemble implementation.

This module provides NMS and Soft-NMS based ensembling for combining
predictions from multiple detection models.

Example:
    >>> from objdet.models.ensemble import NMSEnsemble
    >>>
    >>> ensemble = NMSEnsemble(
    ...     models=[model1, model2],
    ...     iou_thresh=0.5,
    ...     soft_nms=True,
    ... )
    >>> predictions = ensemble(images)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from ensemble_boxes import nms, soft_nms

from objdet.core.constants import EnsembleStrategy
from objdet.core.logging import get_logger
from objdet.models.ensemble.base import BaseEnsemble
from objdet.models.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction

logger = get_logger(__name__)


@MODEL_REGISTRY.register("nms_ensemble")
class NMSEnsemble(BaseEnsemble):
    """NMS-based ensemble.

    Combines predictions from multiple models using Non-Maximum Suppression.
    All predictions are pooled together and standard NMS is applied.

    Args:
        models: List of detector models to ensemble.
        iou_thresh: IoU threshold for NMS.
        soft_nms: If True, use Soft-NMS instead of standard NMS.
        soft_nms_sigma: Sigma parameter for Soft-NMS Gaussian penalty.
        soft_nms_method: Soft-NMS method - "gaussian" or "linear".
        **kwargs: Additional arguments for BaseEnsemble.

    Example:
        >>> # Standard NMS ensemble
        >>> ensemble = NMSEnsemble(models=[m1, m2], iou_thresh=0.5)
        >>>
        >>> # Soft-NMS ensemble
        >>> ensemble = NMSEnsemble(
        ...     models=[m1, m2],
        ...     soft_nms=True,
        ...     soft_nms_sigma=0.5,
        ... )
    """

    strategy = EnsembleStrategy.NMS

    def __init__(
        self,
        models: list,
        iou_thresh: float = 0.5,
        soft_nms: bool = False,
        soft_nms_sigma: float = 0.5,
        soft_nms_method: str = "gaussian",
        **kwargs: Any,
    ) -> None:
        self.use_soft_nms = soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_method = soft_nms_method

        if soft_nms:
            self.strategy = EnsembleStrategy.SOFT_NMS

        super().__init__(
            models=models,
            iou_thresh=iou_thresh,
            **kwargs,
        )

    def _fuse_predictions(
        self,
        predictions: list[DetectionPrediction],
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        """Fuse predictions using NMS or Soft-NMS.

        Args:
            predictions: List of predictions from each model.
            image_size: (height, width) of the image.

        Returns:
            Fused prediction dictionary.
        """
        height, width = image_size

        # Prepare inputs (normalized coordinates)
        boxes_list = []
        scores_list = []
        labels_list = []

        for pred in predictions:
            if pred["boxes"].numel() == 0:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue

            # Normalize boxes to [0, 1]
            boxes = pred["boxes"].clone()
            boxes[:, [0, 2]] /= width
            boxes[:, [1, 3]] /= height
            boxes = boxes.clamp(0, 1)

            boxes_list.append(boxes.cpu().numpy().tolist())
            scores_list.append(pred["scores"].cpu().numpy().tolist())
            labels_list.append(pred["labels"].cpu().numpy().tolist())

        # Check if any predictions exist
        if all(len(b) == 0 for b in boxes_list):
            return {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            }

        # Apply NMS or Soft-NMS
        if self.use_soft_nms:
            fused_boxes, fused_scores, fused_labels = soft_nms(
                boxes_list,
                scores_list,
                labels_list,
                weights=self.weights,
                iou_thr=self.iou_thresh,
                sigma=self.soft_nms_sigma,
                thresh=self.conf_thresh,
                method=2 if self.soft_nms_method == "gaussian" else 1,
            )
        else:
            fused_boxes, fused_scores, fused_labels = nms(
                boxes_list,
                scores_list,
                labels_list,
                weights=self.weights,
                iou_thr=self.iou_thresh,
            )

        # Convert back to pixel coordinates
        fused_boxes = torch.tensor(fused_boxes, dtype=torch.float32)
        if fused_boxes.numel() > 0:
            fused_boxes[:, [0, 2]] *= width
            fused_boxes[:, [1, 3]] *= height

        return {
            "boxes": fused_boxes,
            "labels": torch.tensor(fused_labels, dtype=torch.int64),
            "scores": torch.tensor(fused_scores, dtype=torch.float32),
        }
