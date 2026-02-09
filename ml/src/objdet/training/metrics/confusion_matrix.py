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

    is_differentiable: bool | None = False
    higher_is_better: bool | None = None
    full_state_update: bool | None = False
    matrix: Tensor  # Declared by add_state

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
        )  # type: ignore[assignment]

    def update(  # type: ignore
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
        from torchvision.ops import box_iou

        pred_boxes, pred_labels = self._filter_predictions(pred)
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        if len(gt_boxes) == 0:
            self._update_false_positives(pred_labels)
            return

        if len(pred_boxes) == 0:
            self._update_false_negatives(gt_labels)
            return

        iou_matrix = box_iou(pred_boxes, gt_boxes)
        valid_mask = iou_matrix >= self.iou_threshold

        if not valid_mask.any():
            self._update_false_positives(pred_labels)
            self._update_false_negatives(gt_labels)
            return

        self._process_matches(
            pred_boxes,
            pred_labels,
            gt_boxes,
            gt_labels,
            iou_matrix,
            valid_mask,
        )

    def _filter_predictions(self, pred: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Filter predictions based on confidence threshold."""
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        keep = pred_scores >= self.confidence_threshold
        return pred_boxes[keep], pred_labels[keep]

    def _update_false_positives(self, pred_labels: Tensor) -> None:
        """Mark all predictions as false positives."""
        for p_label in pred_labels:
            self.matrix[self.num_classes, p_label] += 1

    def _update_false_negatives(self, gt_labels: Tensor) -> None:
        """Mark all ground truths as false negatives."""
        for gt_label in gt_labels:
            self.matrix[gt_label, self.num_classes] += 1

    def _process_matches(
        self,
        pred_boxes: Tensor,
        pred_labels: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        iou_matrix: Tensor,
        valid_mask: Tensor,
    ) -> None:
        """Process matches greedily based on IoU."""
        pairs = torch.nonzero(valid_mask)
        pred_indices = pairs[:, 0]
        gt_indices = pairs[:, 1]

        # Sort by IoU descending
        img_ious = iou_matrix[pred_indices, gt_indices]
        sorted_indices = torch.argsort(img_ious, descending=True)
        pred_indices = pred_indices[sorted_indices]
        gt_indices = gt_indices[sorted_indices]

        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
        pred_matched = torch.zeros(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device)

        for p_idx, g_idx in zip(pred_indices, gt_indices, strict=True):
            p_idx_item = p_idx.item()
            g_idx_item = g_idx.item()

            if gt_matched[g_idx_item] or pred_matched[p_idx_item]:
                continue

            gt_matched[g_idx_item] = True
            pred_matched[p_idx_item] = True

            gt_cls = gt_labels[g_idx_item]
            pred_cls = pred_labels[p_idx_item]
            self.matrix[gt_cls, pred_cls] += 1

        # Handle unmatched
        unmatched_preds_mask = ~pred_matched
        self._update_false_positives(pred_labels[unmatched_preds_mask])

        unmatched_gt_mask = ~gt_matched
        self._update_false_negatives(gt_labels[unmatched_gt_mask])

    def compute(self) -> Tensor:
        """Compute confusion matrix.

        Returns:
            Confusion matrix tensor of shape (num_classes+1, num_classes+1).
        """
        return self.matrix
