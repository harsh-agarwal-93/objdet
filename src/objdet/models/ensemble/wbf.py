"""Weighted Box Fusion (WBF) ensemble implementation.

WBF is a method for combining bounding box predictions from multiple
models that produces more accurate boxes than standard NMS by averaging
overlapping predictions rather than suppressing them.

Reference:
    https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Example:
    >>> from objdet.models.ensemble import WBFEnsemble
    >>>
    >>> ensemble = WBFEnsemble(
    ...     models=[model1, model2, model3],
    ...     iou_thresh=0.55,
    ...     weights=[0.5, 0.3, 0.2],
    ... )
    >>> predictions = ensemble(images)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from ensemble_boxes import weighted_boxes_fusion

from objdet.core.constants import EnsembleStrategy
from objdet.core.logging import get_logger
from objdet.models.ensemble.base import BaseEnsemble
from objdet.models.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction

logger = get_logger(__name__)


@MODEL_REGISTRY.register("wbf_ensemble", aliases=["wbf"])
class WBFEnsemble(BaseEnsemble):
    """Weighted Box Fusion ensemble.

    WBF combines predictions by:
    1. Clustering overlapping boxes from all models
    2. Computing weighted average of box coordinates within each cluster
    3. Computing weighted average of confidence scores

    This produces more accurate box locations than NMS since information
    from all models is preserved rather than discarded.

    Args:
        models: List of detector models to ensemble.
        iou_thresh: IoU threshold for box matching.
        skip_box_thresh: Skip boxes with confidence below this threshold.
        conf_type: How to calculate confidence of fused box:
            - "avg": Average of matched boxes
            - "max": Maximum of matched boxes
            - "box_and_model_avg": Average of box and model confidences
            - "absent_model_aware_avg": Handle absent model predictions
        **kwargs: Additional arguments for BaseEnsemble.

    Example:
        >>> # Create ensemble with custom weights
        >>> ensemble = WBFEnsemble(
        ...     models=[faster_rcnn, yolov8, retinanet],
        ...     weights=[0.4, 0.35, 0.25],
        ...     iou_thresh=0.55,
        ... )
    """

    strategy = EnsembleStrategy.WBF

    def __init__(
        self,
        models: list,
        iou_thresh: float = 0.55,
        skip_box_thresh: float = 0.0,
        conf_type: str = "avg",
        **kwargs: Any,
    ) -> None:
        self.skip_box_thresh = skip_box_thresh
        self.conf_type = conf_type

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
        """Fuse predictions using Weighted Box Fusion.

        Args:
            predictions: List of predictions from each model.
            image_size: (height, width) of the image.

        Returns:
            Fused prediction dictionary.
        """
        height, width = image_size

        # Prepare inputs for WBF (normalized coordinates)
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

        # Apply WBF
        if all(len(b) == 0 for b in boxes_list):
            # No predictions from any model
            return {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            }

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=self.weights,
            iou_thr=self.iou_thresh,
            skip_box_thr=self.skip_box_thresh,
            conf_type=self.conf_type,
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
