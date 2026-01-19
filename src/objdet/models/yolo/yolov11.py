"""YOLOv11 implementation as a Lightning module.

This module wraps the Ultralytics YOLOv11 (YOLO11) model for use with
PyTorch Lightning.

Example:
    >>> from objdet.models.yolo import YOLOv11
    >>>
    >>> model = YOLOv11(num_classes=80, model_size="m", pretrained=True)
    >>> trainer = Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
"""

from __future__ import annotations

from typing import Any

from objdet.core.logging import get_logger
from objdet.models.registry import MODEL_REGISTRY
from objdet.models.yolo.base import YOLOBaseLightning

logger = get_logger(__name__)


@MODEL_REGISTRY.register("yolov11", aliases=["yolo11"])
class YOLOv11(YOLOBaseLightning):
    """YOLOv11 (YOLO11) object detection model wrapped for Lightning.

    YOLOv11 is the latest iteration of the YOLO series featuring:
    - Improved C3k2 blocks for better feature extraction
    - Enhanced attention mechanisms
    - Better small object detection
    - Optimized architecture for efficiency

    Available model sizes:
    - n (nano): Fastest, lowest accuracy
    - s (small): Fast with good accuracy
    - m (medium): Balanced speed/accuracy
    - l (large): High accuracy
    - x (extra-large): Highest accuracy

    Args:
        num_classes: Number of object classes (no background).
        model_size: Model size variant ("n", "s", "m", "l", "x").
        pretrained: If True, load COCO pretrained weights.
        conf_thres: Confidence threshold for predictions.
        iou_thres: IoU threshold for NMS.
        **kwargs: Additional arguments for BaseLightningDetector.

    Example:
        >>> # Create YOLOv11-large model
        >>> model = YOLOv11(num_classes=20, model_size="l")
        >>>
        >>> # Train with Lightning
        >>> trainer = Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule)
    """

    MODEL_VARIANTS = {
        "n": "yolo11n.pt",
        "s": "yolo11s.pt",
        "m": "yolo11m.pt",
        "l": "yolo11l.pt",
        "x": "yolo11x.pt",
    }

    def __init__(
        self,
        num_classes: int,
        model_size: str = "n",
        pretrained: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            model_size=model_size,
            pretrained=pretrained,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            **kwargs,
        )

        logger.info(
            f"Created YOLOv11-{model_size}: num_classes={num_classes}, pretrained={pretrained}"
        )

    def _get_model_variant(self) -> str:
        """Get the YOLOv11 model variant string.

        Returns:
            Model variant string (e.g., "yolo11m.pt").
        """
        return self.MODEL_VARIANTS[self.model_size]
