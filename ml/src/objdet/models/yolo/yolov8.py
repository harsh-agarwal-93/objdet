"""YOLOv8 implementation as a Lightning module.

This module wraps the Ultralytics YOLOv8 model for use with PyTorch Lightning.

Example:
    >>> from objdet.models.yolo import YOLOv8
    >>>
    >>> model = YOLOv8(num_classes=80, model_size="m", pretrained=True)
    >>> trainer = Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
"""

from __future__ import annotations

from typing import Any

from objdet.core.logging import get_logger
from objdet.models.registry import MODEL_REGISTRY
from objdet.models.yolo.base import YOLOBaseLightning

logger = get_logger(__name__)


@MODEL_REGISTRY.register("yolov8", aliases=["yolo8"])
class YOLOv8(YOLOBaseLightning):
    """YOLOv8 object detection model wrapped for Lightning.

    YOLOv8 is a state-of-the-art real-time object detector featuring:
    - Anchor-free detection head
    - C2f modules for efficient feature extraction
    - Mosaic and MixUp augmentation (handled via transforms)
    - Task-aligned assigner for positive sample selection

    Available model sizes:
    - n (nano): Fastest, lowest accuracy (~3.2M params)
    - s (small): Fast with good accuracy (~11.2M params)
    - m (medium): Balanced speed/accuracy (~25.9M params)
    - l (large): High accuracy (~43.7M params)
    - x (extra-large): Highest accuracy (~68.2M params)

    Warning:
        There is a known bug in the training pipeline that causes
        ``IndexError: too many indices for tensor of dimension 2``
        during the loss computation. This affects training via both CLI
        and Python API. Investigation is ongoing to resolve this issue
        in the Ultralytics loss integration.

    Args:
        num_classes: Number of object classes (no background).
        model_size: Model size variant ("n", "s", "m", "l", "x").
        pretrained: If True, load COCO pretrained weights.
        conf_thres: Confidence threshold for predictions.
        iou_thres: IoU threshold for NMS.
        **kwargs: Additional arguments for BaseLightningDetector.

    Example:
        >>> # Create YOLOv8-medium model
        >>> model = YOLOv8(num_classes=20, model_size="m")
        >>>
        >>> # Train with Lightning
        >>> trainer = Trainer(
        ...     max_epochs=100,
        ...     callbacks=[ModelCheckpoint(monitor="val/mAP")],
        ... )
        >>> trainer.fit(model, datamodule)
    """

    MODEL_VARIANTS = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
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
            f"Created YOLOv8-{model_size}: num_classes={num_classes}, pretrained={pretrained}"
        )

    def _get_model_variant(self) -> str:
        """Get the YOLOv8 model variant string.

        Returns:
            Model variant string (e.g., "yolov8m.pt").
        """
        return self.MODEL_VARIANTS[self.model_size]
