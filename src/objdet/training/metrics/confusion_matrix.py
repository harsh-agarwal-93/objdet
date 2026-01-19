"""Confusion matrix metric for object detection.

A TorchMetrics-compatible confusion matrix implementation for
object detection tasks.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

from objdet.core.logging import get_logger

logger = get_logger(__name__)


class ConfusionMatrix(Metric):
    """Confusion matrix metric for object detection.

    Unlike classification confusion matrices, detection requires
    matching predictions to ground truth using IoU threshold.

    Args:
        num_classes: Number of object classes.
        iou_threshold: IoU threshold for matching.
        confidence_threshold: Minimum prediction confidence.
        dist_sync_on_step: Whether to sync on step (for DDP).

    Example:
        >>> metric = ConfusionMatrix(num_classes=80)
        >>> metric.update(predictions, targets)
        >>> cm = metric.compute()
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.25,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        # State: confusion matrix
        # +1 for background (no match)
        self.add_state(
            "matrix",
            default=torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> None:
        """Update confusion matrix with batch of predictions.

        Args:
            preds: List of prediction dicts with boxes, labels, scores.
            targets: List of target dicts with boxes, labels.
        """
        for pred, target in zip(preds, targets, strict=True):
            self._update_single(pred, target)

    def _update_single(
        self,
        pred: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> None:
        """Update with single image predictions."""
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Filter predictions by confidence
        mask = pred_scores >= self.confidence_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        # Track matched GT
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

        # Match predictions
        for p_box, p_label in zip(pred_boxes, pred_labels, strict=True):
            best_iou = 0.0
            best_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if gt_matched[idx]:
                    continue

                iou = self._box_iou(p_box, gt_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                gt_matched[best_idx] = True
                self.matrix[gt_labels[best_idx], p_label] += 1
            else:
                # False positive
                self.matrix[self.num_classes, p_label] += 1

        # False negatives (missed GT)
        for idx, (matched, gt_label) in enumerate(zip(gt_matched, gt_labels, strict=True)):
            if not matched:
                self.matrix[gt_label, self.num_classes] += 1

    def _box_iou(self, box1: Tensor, box2: Tensor) -> float:
        """Compute IoU between two boxes."""
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        if union == 0:
            return 0.0

        return (inter / union).item()

    def compute(self) -> Tensor:
        """Compute confusion matrix.

        Returns:
            Confusion matrix tensor of shape (num_classes+1, num_classes+1).
        """
        return self.matrix
