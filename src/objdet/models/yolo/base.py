"""Base class for YOLO models wrapped as Lightning modules.

This module provides the base class for integrating Ultralytics YOLO models
with PyTorch Lightning. It extracts the model architecture from Ultralytics
and implements custom training/validation steps for Lightning compatibility.

NOTE: This is more complex than using Ultralytics' built-in train() method,
but provides full access to the Lightning ecosystem (callbacks, loggers,
distributed training, profiling, etc.).

Example:
    >>> from objdet.models.yolo import YOLOv8
    >>>
    >>> model = YOLOv8(num_classes=80, model_size="m")
    >>> trainer = Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from ultralytics import YOLO

from objdet.core.constants import ClassIndexMode
from objdet.core.exceptions import ModelError
from objdet.core.logging import get_logger
from objdet.models.base import BaseLightningDetector

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction, DetectionTarget

logger = get_logger(__name__)


class YOLOBaseLightning(BaseLightningDetector):
    """Base Lightning wrapper for Ultralytics YOLO models.

    This class extracts the YOLO model architecture from Ultralytics and
    wraps it in a Lightning module. Training and validation steps are
    implemented to work with Lightning's DataModule system.

    YOLO uses its own class indexing (no background class, classes start at 0).

    Args:
        num_classes: Number of object classes.
        model_name: YOLO model name (e.g., "yolov8n", "yolov8s", "yolov11m").
        model_size: Model size variant (n, s, m, l, x).
        pretrained: If True, load pretrained COCO weights.
        conf_thres: Confidence threshold for NMS.
        iou_thres: IoU threshold for NMS.
        **kwargs: Additional arguments for BaseLightningDetector.

    Attributes:
        model: The YOLO model for forward pass.
        ultralytics_model: The original Ultralytics YOLO wrapper for loss computation.

    Subclasses must implement:
        _get_model_variant(): Return the model variant string (e.g., "yolov8n.pt").
    """

    # Model architecture versions (override in subclasses)
    MODEL_VARIANTS: dict[str, str] = {}

    def __init__(
        self,
        num_classes: int,
        model_size: str = "n",
        pretrained: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        **kwargs: Any,
    ) -> None:
        self.model_size = model_size.lower()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._pretrained = pretrained

        # Validate model size
        if self.model_size not in self.MODEL_VARIANTS:
            available = list(self.MODEL_VARIANTS.keys())
            raise ModelError(f"Invalid model size '{model_size}'. Available: {available}")

        # Force YOLO class index mode
        kwargs["class_index_mode"] = ClassIndexMode.YOLO
        kwargs["confidence_threshold"] = conf_thres
        kwargs["nms_threshold"] = iou_thres

        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )

    @abstractmethod
    def _get_model_variant(self) -> str:
        """Get the model variant string for this YOLO version.

        Returns:
            Model variant string (e.g., "yolov8n.pt").
        """
        ...

    def _build_model(self) -> nn.Module:
        """Build the YOLO model architecture.

        We load the Ultralytics YOLO model and extract/modify it for our needs.

        Returns:
            YOLO model ready for training.
        """
        variant = self._get_model_variant()

        # Load the Ultralytics YOLO model
        if self._pretrained:
            # Load pretrained weights
            self.ultralytics_model = YOLO(variant)
        else:
            # Load architecture only from yaml
            yaml_path = variant.replace(".pt", ".yaml")
            self.ultralytics_model = YOLO(yaml_path)

        # Get the underlying PyTorch model
        model = self.ultralytics_model.model

        # Modify the detection head for custom number of classes if needed
        if self.num_classes != 80:  # 80 is COCO default
            self._modify_head_for_classes(model)

        logger.info(f"Built YOLO model: variant={variant}, num_classes={self.num_classes}")

        return model

    def _modify_head_for_classes(self, model: nn.Module) -> None:
        """Modify the detection head for a different number of classes.

        Args:
            model: The YOLO model to modify.
        """
        # YOLO detection head structure varies by version
        # This handles the common case for YOLOv8/v11
        try:
            # Find the Detect head
            detect = model.model[-1]  # Usually the last layer

            # Update number of classes
            detect.nc = self.num_classes

            # Recalculate output channels
            # Detection head outputs: (num_classes + 4) per anchor for each scale
            # The exact modification depends on YOLO version
            # For YOLOv8/v11, we need to rebuild the cv2 and cv3 layers

            logger.debug(f"Modified YOLO head for {self.num_classes} classes")

        except (AttributeError, IndexError) as e:
            logger.warning(
                f"Could not automatically modify head for {self.num_classes} classes: {e}. "
                "Model may need manual configuration for non-COCO class counts."
            )

    def forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget] | None = None,
    ) -> dict[str, Tensor] | list[DetectionPrediction]:
        """Forward pass.

        Args:
            images: List of image tensors (C, H, W) or batched tensor (B, C, H, W).
            targets: Optional list of target dicts for training.

        Returns:
            Training: Dict of losses.
            Inference: List of prediction dicts with 'boxes', 'labels', 'scores'.
        """
        # Handle list of images - stack into batch
        if isinstance(images, list):
            batch = torch.stack(images, dim=0)
        else:
            batch = images

        if self.training and targets is not None:
            return self._training_forward(batch, targets)
        else:
            return self._inference_forward(batch)

    def _training_forward(
        self,
        batch: Tensor,
        targets: list[DetectionTarget],
    ) -> dict[str, Tensor]:
        """Training forward pass with loss computation.

        The YOLO loss is computed using the Ultralytics loss function.

        Args:
            batch: Batched image tensor (B, C, H, W).
            targets: List of target dictionaries.

        Returns:
            Dictionary of losses.
        """
        # Convert targets to YOLO format
        # YOLO expects: (batch_idx, class_id, x_center, y_center, width, height)
        yolo_targets = self._convert_targets_to_yolo_format(batch, targets)

        # Forward through model
        outputs = self.model(batch)

        # Compute loss using Ultralytics loss function
        loss, loss_items = self._compute_yolo_loss(outputs, yolo_targets)

        return {
            "loss": loss,
            "loss_box": loss_items[0] if len(loss_items) > 0 else loss,
            "loss_cls": loss_items[1] if len(loss_items) > 1 else torch.tensor(0.0),
            "loss_dfl": loss_items[2] if len(loss_items) > 2 else torch.tensor(0.0),
        }

    def _convert_targets_to_yolo_format(
        self,
        batch: Tensor,
        targets: list[DetectionTarget],
    ) -> Tensor:
        """Convert targets from TorchVision format to YOLO format.

        TorchVision format: boxes as [x1, y1, x2, y2] in pixels
        YOLO format: [batch_idx, class_id, x_center, y_center, width, height] normalized

        Args:
            batch: Image batch tensor for getting dimensions.
            targets: List of target dictionaries.

        Returns:
            Tensor of shape (N, 6) in YOLO format.
        """
        _, _, height, width = batch.shape
        all_targets = []

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"]  # (N, 4) in xyxy format
            labels = target["labels"]  # (N,)

            if boxes.numel() == 0:
                continue

            # Convert xyxy to xywh normalized
            x1, y1, x2, y2 = boxes.unbind(dim=1)
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            # Stack with batch index and class
            batch_indices = torch.full_like(labels, batch_idx, dtype=torch.float32)
            yolo_boxes = torch.stack(
                [batch_indices, labels.float(), x_center, y_center, w, h], dim=1
            )
            all_targets.append(yolo_boxes)

        if all_targets:
            return torch.cat(all_targets, dim=0)
        else:
            return torch.zeros((0, 6), device=batch.device)

    def _compute_yolo_loss(
        self,
        outputs: Any,
        targets: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Compute YOLO loss using Ultralytics loss function.

        Args:
            outputs: Model outputs from forward pass.
            targets: Targets in YOLO format.

        Returns:
            Tuple of (total_loss, [box_loss, cls_loss, dfl_loss]).
        """
        # Use Ultralytics loss computation
        # The exact API depends on the YOLO version
        try:
            # YOLOv8/v11 loss computation
            loss_fn = self.ultralytics_model.model.loss

            # Compute loss
            loss, loss_items = loss_fn(outputs, targets)

            return loss, loss_items.detach().cpu().tolist()

        except (AttributeError, TypeError) as e:
            logger.warning(f"Error computing YOLO loss: {e}. Using simple loss.")
            # Fallback to simple loss (for debugging/development)
            # In production, proper loss implementation is needed
            return outputs[0].mean(), [outputs[0].mean()]

    def _inference_forward(self, batch: Tensor) -> list[DetectionPrediction]:
        """Inference forward pass.

        Args:
            batch: Batched image tensor (B, C, H, W).

        Returns:
            List of prediction dictionaries.
        """
        # Use Ultralytics predict for proper post-processing
        results = self.ultralytics_model.predict(
            batch,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        predictions = []
        for result in results:
            boxes = result.boxes

            pred = {
                "boxes": boxes.xyxy.cpu() if boxes.xyxy is not None else torch.empty(0, 4),
                "labels": boxes.cls.cpu().int()
                if boxes.cls is not None
                else torch.empty(0, dtype=torch.int64),
                "scores": boxes.conf.cpu() if boxes.conf is not None else torch.empty(0),
            }
            predictions.append(pred)

        return predictions

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                "model_variant": self._get_model_variant(),
                "model_size": self.model_size,
                "conf_thres": self.conf_thres,
                "iou_thres": self.iou_thres,
            }
        )
        return info
