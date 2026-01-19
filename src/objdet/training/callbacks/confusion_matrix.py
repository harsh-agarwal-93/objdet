"""Confusion matrix callback for object detection.

This callback computes and saves a confusion matrix at the end of
each validation epoch, useful for analyzing class-wise performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ConfusionMatrixCallback(Callback):
    """Callback to compute and save confusion matrix.

    The confusion matrix shows true vs predicted classes for all
    detections matched using IoU threshold.

    Args:
        num_classes: Number of object classes (not including background).
        iou_threshold: IoU threshold for matching predictions to ground truth.
        confidence_threshold: Minimum confidence for predictions.
        save_dir: Directory to save confusion matrix plots.
        class_names: Optional list of class names for axis labels.
        normalize: How to normalize - "true", "pred", "all", or None.
        save_format: File format for saving ("png", "pdf", "svg").

    Example:
        >>> callback = ConfusionMatrixCallback(
        ...     num_classes=80,
        ...     save_dir="outputs/confusion_matrices",
        ...     class_names=["person", "car", ...],
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.25,
        save_dir: str | Path = "outputs/confusion_matrices",
        class_names: list[str] | None = None,
        normalize: str | None = "true",
        save_format: str = "png",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.save_dir = Path(save_dir)
        self.class_names = class_names
        self.normalize = normalize
        self.save_format = save_format

        # Confusion matrix accumulator: [true_class, pred_class]
        # +1 for background predictions (no match)
        self._confusion_matrix: Tensor | None = None

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Reset confusion matrix at start of validation."""
        # +1 for "background" (false positives / no GT match)
        self._confusion_matrix = torch.zeros(
            (self.num_classes + 1, self.num_classes + 1),
            dtype=torch.int64,
            device=pl_module.device,
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update confusion matrix with batch predictions."""
        if self._confusion_matrix is None:
            return

        # Get predictions from model
        images, targets = batch
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module(images)

        # Update confusion matrix for each image
        for pred, target in zip(predictions, targets, strict=True):
            self._update_matrix(pred, target)

    def _update_matrix(
        self,
        pred: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> None:
        """Update confusion matrix with single image predictions."""
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Filter by confidence
        mask = pred_scores >= self.confidence_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        # Track which GT boxes have been matched
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

        # Match predictions to ground truth
        for p_box, p_label in zip(pred_boxes, pred_labels, strict=True):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels, strict=True)):
                if gt_matched[gt_idx]:
                    continue

                iou = self._compute_iou(p_box, gt_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                # True positive: matched GT
                gt_matched[best_gt_idx] = True
                true_class = gt_labels[best_gt_idx].item()
                pred_class = p_label.item()
            else:
                # False positive: no GT match (background)
                true_class = self.num_classes  # Background index
                pred_class = p_label.item()

            self._confusion_matrix[true_class, pred_class] += 1

        # False negatives: unmatched GT
        for gt_idx, (matched, gt_label) in enumerate(zip(gt_matched, gt_labels, strict=True)):
            if not matched:
                true_class = gt_label.item()
                pred_class = self.num_classes  # Not detected
                self._confusion_matrix[true_class, pred_class] += 1

    def _compute_iou(self, box1: Tensor, box2: Tensor) -> float:
        """Compute IoU between two boxes in xyxy format."""
        x1 = max(box1[0].item(), box2[0].item())
        y1 = max(box1[1].item(), box2[1].item())
        x2 = min(box1[2].item(), box2[2].item())
        y2 = min(box1[3].item(), box2[3].item())

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area.item() + box2_area.item() - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Save confusion matrix at end of validation."""
        if self._confusion_matrix is None:
            return

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy for later analysis
        save_path = self.save_dir / f"confusion_matrix_epoch_{trainer.current_epoch}.pt"
        torch.save(self._confusion_matrix.cpu(), save_path)

        # Try to save visualization
        try:
            self._save_plot(trainer.current_epoch)
        except ImportError:
            logger.warning("matplotlib not available for confusion matrix plotting")

        logger.info(f"Saved confusion matrix to {save_path}")

    def _save_plot(self, epoch: int) -> None:
        """Save confusion matrix as a plot."""
        import matplotlib.pyplot as plt
        import numpy as np

        cm = self._confusion_matrix.cpu().numpy()

        # Normalize if requested
        if self.normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif self.normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif self.normalize == "all":
            cm = cm / cm.sum()

        # Replace NaN with 0
        cm = np.nan_to_num(cm)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap="Blues")

        # Labels
        if self.class_names:
            labels = self.class_names + ["background"]
        else:
            labels = [str(i) for i in range(self.num_classes)] + ["bg"]

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")

        plt.colorbar(im)
        plt.tight_layout()

        save_path = self.save_dir / f"confusion_matrix_epoch_{epoch}.{self.save_format}"
        plt.savefig(save_path, dpi=150)
        plt.close()
