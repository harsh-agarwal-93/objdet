"""RetinaNet implementation as a Lightning module.

This module wraps the TorchVision RetinaNet model for use with
PyTorch Lightning.

Example:
    >>> from objdet.models.torchvision import RetinaNet
    >>>
    >>> model = RetinaNet(num_classes=80, pretrained_backbone=True)
    >>> predictions = model([torch.rand(3, 800, 600)])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from objdet.core.constants import ClassIndexMode
from objdet.core.logging import get_logger
from objdet.models.base import BaseLightningDetector
from objdet.models.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction, DetectionTarget

logger = get_logger(__name__)


@MODEL_REGISTRY.register("retinanet")
class RetinaNet(BaseLightningDetector):
    """RetinaNet with ResNet-50 FPN backbone.

    RetinaNet is a one-stage object detector that uses:
    1. Feature Pyramid Network (FPN) for multi-scale features
    2. Focal loss to address class imbalance
    3. Separate classification and regression heads

    The model uses TorchVision class indexing (background at index 0).

    Args:
        num_classes: Number of object classes (NOT including background).
        backbone: Backbone variant - "resnet50_fpn" or "resnet50_fpn_v2".
        pretrained: If True, use pretrained weights on COCO.
        pretrained_backbone: If True, use ImageNet pretrained backbone.
        trainable_backbone_layers: Number of trainable backbone layers (0-5).
        min_size: Minimum image size for inference.
        max_size: Maximum image size for inference.
        score_thresh: Score threshold for predictions.
        nms_thresh: NMS threshold.
        detections_per_img: Maximum detections per image.
        **kwargs: Additional arguments for BaseLightningDetector.

    Example:
        >>> model = RetinaNet(num_classes=20, pretrained_backbone=True)
        >>> trainer = Trainer(max_epochs=50)
        >>> trainer.fit(model, datamodule)
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
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        **kwargs: Any,
    ) -> None:
        self.backbone_name = backbone
        self.trainable_backbone_layers = trainable_backbone_layers
        self.min_size = min_size
        self.max_size = max_size
        self.score_thresh = score_thresh
        self.nms_thresh_model = nms_thresh
        self.detections_per_img = detections_per_img
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
            f"Created RetinaNet: backbone={backbone}, num_classes={num_classes} (+1 background)"
        )

    def _build_model(self) -> nn.Module:
        """Build the RetinaNet model architecture.

        Returns:
            Configured RetinaNet model.
        """
        # Determine weights
        if self._pretrained_coco:
            if self.backbone_name == "resnet50_fpn_v2":
                weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
            else:
                weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
            weights_backbone = None
        elif self.pretrained_backbone:
            weights = None
            weights_backbone = "IMAGENET1K_V1"
        else:
            weights = None
            weights_backbone = None

        # Build model
        if self.backbone_name == "resnet50_fpn_v2":
            model = retinanet_resnet50_fpn_v2(
                weights=weights,
                weights_backbone=weights_backbone,
                trainable_backbone_layers=self.trainable_backbone_layers,
                min_size=self.min_size,
                max_size=self.max_size,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh_model,
                detections_per_img=self.detections_per_img,
            )
        else:
            model = retinanet_resnet50_fpn(
                weights=weights,
                weights_backbone=weights_backbone,
                trainable_backbone_layers=self.trainable_backbone_layers,
                min_size=self.min_size,
                max_size=self.max_size,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh_model,
                detections_per_img=self.detections_per_img,
            )

        # Replace classification head for custom number of classes
        # RetinaNet head uses num_classes directly (no background)
        # But TorchVision internally adds 1 for background
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.head.classification_head.conv[0].in_channels

        # Create new classification head
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=self.num_model_classes,  # Includes background
        )

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
            Training: Dict of losses {'classification', 'bbox_regression'}.
            Inference: List of prediction dicts with 'boxes', 'labels', 'scores'.
        """
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
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
                "score_thresh": self.score_thresh,
                "detections_per_img": self.detections_per_img,
            }
        )
        return info
