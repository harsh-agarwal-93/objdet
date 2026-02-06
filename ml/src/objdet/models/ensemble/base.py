"""Base class for ensemble detection models.

This module provides the base class for combining predictions from
multiple detection models into a single ensemble prediction.

Example:
    >>> from objdet.models.ensemble import WBFEnsemble
    >>>
    >>> # Create ensemble from multiple models
    >>> models = [faster_rcnn, yolov8]
    >>> ensemble = WBFEnsemble(models=models, iou_thresh=0.5)
    >>>
    >>> # Run ensemble inference
    >>> predictions = ensemble(images)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from objdet.core.constants import ClassIndexMode, EnsembleStrategy
from objdet.core.logging import get_logger
from objdet.models.base import BaseLightningDetector

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction, DetectionTarget

logger = get_logger(__name__)


class BaseEnsemble(BaseLightningDetector):
    """Abstract base class for ensemble detection models.

    An ensemble combines predictions from multiple detection models using
    various fusion strategies (WBF, NMS, etc.).

    Args:
        models: List of detector models to ensemble.
        iou_thresh: IoU threshold for box fusion/suppression.
        conf_thresh: Minimum confidence threshold for predictions.
        weights: Optional weights for each model (must sum to 1 if provided).
        class_index_mode: Class index mode for the ensemble output.
        **kwargs: Additional arguments for BaseLightningDetector.

    Attributes:
        models: List of component models.
        weights: Model weights for weighted fusion.
        strategy: Ensemble strategy enum.

    Note:
        All models must have compatible class definitions (same classes
        in the same order).
    """

    strategy: EnsembleStrategy = EnsembleStrategy.NMS

    def __init__(
        self,
        models: list[BaseLightningDetector | nn.Module],
        iou_thresh: float = 0.5,
        conf_thresh: float = 0.25,
        weights: list[float] | None = None,
        class_index_mode: ClassIndexMode = ClassIndexMode.YOLO,
        **kwargs: Any,
    ) -> None:
        # Get num_classes from first model
        if hasattr(models[0], "num_classes"):
            num_classes = models[0].num_classes
        else:
            raise ValueError("First model must have num_classes attribute")

        # Validate models have same number of classes
        for i, model in enumerate(models[1:], 1):
            if hasattr(model, "num_classes") and model.num_classes != num_classes:
                raise ValueError(
                    f"Model {i} has {model.num_classes} classes, "
                    f"but model 0 has {num_classes} classes"
                )

        self.ensemble_models = nn.ModuleList(models)
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

        # Set weights (default to equal weights)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(models)})"
                )
            # Normalize weights if they don't sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]

        super().__init__(
            num_classes=cast("int", num_classes),
            class_index_mode=class_index_mode,
            confidence_threshold=conf_thresh,
            nms_threshold=iou_thresh,
            pretrained=False,
            **kwargs,
        )

        logger.info(
            f"Created {self.__class__.__name__} with {len(models)} models, weights={self.weights}"
        )

    def _build_model(self) -> nn.Module:
        """Return the ensemble as the model.

        The ensemble doesn't have a single model architecture;
        it uses the list of models in self.ensemble_models.
        """
        return nn.Identity()  # Placeholder; actual models in ensemble_models

    def forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget] | None = None,
    ) -> dict[str, Tensor] | list[DetectionPrediction]:
        """Forward pass through ensemble.

        Args:
            images: List of image tensors.
            targets: Optional targets for training (not typically used for ensembles).

        Returns:
            Training: Combined loss (if all models support training).
            Inference: Fused predictions from all models.
        """
        if self.training and targets is not None:
            return self._training_forward(images, targets)
        else:
            return self._inference_forward(images)

    def _training_forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget],
    ) -> dict[str, Tensor]:
        """Training forward (train each model separately).

        Note: Typically ensembles are used for inference only.
        Training is done on individual models.

        Args:
            images: List of image tensors.
            targets: List of target dictionaries.

        Returns:
            Weighted average of losses from all models.
        """
        total_loss = torch.tensor(0.0, device=images[0].device)
        loss_dict: dict[str, Tensor] = {}

        for i, (model, weight) in enumerate(zip(self.ensemble_models, self.weights, strict=True)):
            model_losses = model(images, targets)
            if isinstance(model_losses, dict):
                for key, value in model_losses.items():
                    loss_key = f"model_{i}/{key}"
                    loss_dict[loss_key] = value
                    total_loss = total_loss + weight * value

        loss_dict["loss"] = total_loss
        return loss_dict

    def _inference_forward(
        self,
        images: list[Tensor],
    ) -> list[DetectionPrediction]:
        """Inference forward with prediction fusion.

        Args:
            images: List of image tensors.

        Returns:
            Fused predictions from all models.
        """
        # Get predictions from all models
        all_predictions: list[list[DetectionPrediction]] = []

        for model in self.ensemble_models:
            model.eval()
            with torch.no_grad():
                preds = model(images)
                all_predictions.append(preds)

        # Fuse predictions for each image
        batch_size = len(images)
        fused_predictions = []

        for img_idx in range(batch_size):
            # Collect predictions from all models for this image
            img_preds = [preds[img_idx] for preds in all_predictions]
            fused = self._fuse_predictions(img_preds, images[img_idx].shape[-2:])  # type: ignore[arg-type]
            fused_predictions.append(fused)

        return fused_predictions

    @abstractmethod
    def _fuse_predictions(
        self,
        predictions: list[DetectionPrediction],
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        """Fuse predictions from multiple models.

        Args:
            predictions: List of predictions from each model.
            image_size: (height, width) of the image.

        Returns:
            Fused prediction dictionary.
        """
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get ensemble information."""
        info = super().get_model_info()
        info.update(
            {
                "ensemble_strategy": self.strategy.value,
                "num_models": len(self.ensemble_models),
                "weights": self.weights,
                "iou_thresh": self.iou_thresh,
            }
        )
        return info
