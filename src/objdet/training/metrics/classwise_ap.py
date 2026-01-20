"""Class-wise Average Precision metric.

This metric computes AP per class for detailed performance analysis.
"""

from __future__ import annotations

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision

from objdet.core.logging import get_logger

logger = get_logger(__name__)


class ClasswiseAP(Metric):
    """Compute class-wise Average Precision.

    Wraps torchmetrics MeanAveragePrecision to provide easy
    access to per-class AP values.

    Args:
        num_classes: Number of object classes.
        iou_thresholds: List of IoU thresholds.
        class_names: Optional list of class names.

    Example:
        >>> metric = ClasswiseAP(num_classes=80, class_names=COCO_CLASSES)
        >>> metric.update(predictions, targets)
        >>> ap_per_class = metric.compute()
    """

    is_differentiable: bool | None = False
    higher_is_better: bool | None = True
    full_state_update: bool | None = True

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: list[float] | None = None,
        class_names: list[str] | None = None,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # Use torchmetrics mAP under the hood
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=iou_thresholds,
            class_metrics=True,  # Enable per-class metrics
        )

    def update(
        self,
        preds: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> None:
        """Update metric with batch of predictions.

        Args:
            preds: List of prediction dicts.
            targets: List of target dicts.
        """
        self.map_metric.update(preds, targets)

    def compute(self) -> dict[str, Tensor | dict[str, float]]:
        """Compute class-wise AP.

        Returns:
            Dictionary with overall metrics and per-class AP.
        """
        result = self.map_metric.compute()

        # Extract per-class AP if available
        output = {
            "map": result["map"],
            "map_50": result["map_50"],
            "map_75": result["map_75"],
        }

        # Per-class AP (if available in result)
        if "map_per_class" in result:
            per_class = result["map_per_class"]
            classwise_ap = {}
            for i, ap in enumerate(per_class):
                if i < len(self.class_names):
                    classwise_ap[self.class_names[i]] = ap.item()
            output["classwise_ap"] = classwise_ap

        return output

    def reset(self) -> None:
        """Reset underlying metric."""
        super().reset()
        self.map_metric.reset()
