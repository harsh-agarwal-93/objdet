"""Faster R-CNN implementation as a Lightning module.

This module wraps the TorchVision Faster R-CNN model for use with
PyTorch Lightning, providing a clean interface for training and inference.

Example:
    >>> from objdet.models.torchvision import FasterRCNN
    >>> from lightning import Trainer
    >>>
    >>> model = FasterRCNN(num_classes=80, pretrained_backbone=True)
    >>> trainer = Trainer(max_epochs=100)
    >>> trainer.fit(model, datamodule)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from objdet.core.constants import ClassIndexMode
from objdet.core.logging import get_logger
from objdet.models.base import BaseLightningDetector
from objdet.models.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction, DetectionTarget

logger = get_logger(__name__)


@MODEL_REGISTRY.register("faster_rcnn", aliases=["fasterrcnn", "frcnn"])
class FasterRCNN(BaseLightningDetector):
    """Faster R-CNN with ResNet-50 FPN backbone.

    This is a two-stage object detector consisting of:
    1. Region Proposal Network (RPN) for generating object proposals
    2. Fast R-CNN head for classification and bounding box regression

    The model uses TorchVision class indexing (background at index 0).

    Args:
        num_classes: Number of object classes (NOT including background).
            The model will internally use num_classes + 1.
        backbone: Backbone variant - "resnet50_fpn" or "resnet50_fpn_v2".
        pretrained: If True, use pretrained weights on COCO.
        pretrained_backbone: If True, use ImageNet pretrained backbone.
        trainable_backbone_layers: Number of trainable backbone layers (0-5).
        min_size: Minimum image size for inference.
        max_size: Maximum image size for inference.
        **kwargs: Additional arguments for BaseLightningDetector.

    Attributes:
        model: The underlying TorchVision Faster R-CNN model.

    Example:
        >>> model = FasterRCNN(num_classes=20, pretrained_backbone=True)
        >>> images = [torch.rand(3, 800, 600) for _ in range(4)]
        >>> predictions = model(images)
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50_fpn_v2",
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs: Any,
    ) -> None:
        # Store backbone config before calling super().__init__
        self.backbone_name = backbone
        self.trainable_backbone_layers = trainable_backbone_layers
        self.min_size = min_size
        self.max_size = max_size
        self._pretrained_coco = pretrained

        # Force TorchVision class index mode
        kwargs["class_index_mode"] = ClassIndexMode.TORCHVISION

        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            **kwargs,
        )

        logger.info(
            f"Created Faster R-CNN: backbone={backbone}, num_classes={num_classes} (+1 background)"
        )

    def _build_model(self) -> nn.Module:
        """Build the Faster R-CNN model architecture.

        Returns:
            Configured Faster R-CNN model.
        """
        # Determine weights
        if self._pretrained_coco:
            if self.backbone_name == "resnet50_fpn_v2":
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            else:
                weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            weights_backbone = None  # Use COCO pretrained (includes backbone)
        elif self.pretrained_backbone:
            weights = None
            weights_backbone = "IMAGENET1K_V1"
        else:
            weights = None
            weights_backbone = None

        # Build model
        if self.backbone_name == "resnet50_fpn_v2":
            model = fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=weights_backbone,
                trainable_backbone_layers=self.trainable_backbone_layers,
                min_size=self.min_size,
                max_size=self.max_size,
            )
        else:
            model = fasterrcnn_resnet50_fpn(
                weights=weights,
                weights_backbone=weights_backbone,
                trainable_backbone_layers=self.trainable_backbone_layers,
                min_size=self.min_size,
                max_size=self.max_size,
            )

        # Replace the classifier head for custom number of classes
        # num_model_classes includes background
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_model_classes)

        return model

    def forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget] | None = None,
    ) -> dict[str, Tensor] | list[DetectionPrediction]:
        """Forward pass.

        Args:
            images: List of image tensors (C, H, W).
            targets: Optional list of target dicts for training.

        Returns:
            Training: Dict of losses.
            Inference: List of prediction dicts with boxes, labels, scores.
        """
        if self.training and targets is not None:
            # Training mode: compute losses
            return self.model(images, targets)
        else:
            # Inference mode: get predictions
            self.model.eval()
            with torch.no_grad():
                return self.model(images)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update(
            {
                "backbone": self.backbone_name,
                "trainable_backbone_layers": self.trainable_backbone_layers,
                "min_size": self.min_size,
                "max_size": self.max_size,
            }
        )
        return info
