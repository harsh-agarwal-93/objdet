"""Base class for all object detection models in ObjDet.

This module provides the abstract base class that all detection models
must inherit from. It defines the common interface for training,
validation, and inference.

Example:
    >>> from objdet.models.base import BaseLightningDetector
    >>>
    >>> class MyDetector(BaseLightningDetector):
    ...     def __init__(self, num_classes: int, **kwargs):
    ...         super().__init__(num_classes=num_classes, **kwargs)
    ...         self.model = build_my_model(num_classes)
    ...
    ...     def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
    ...         return self.model(images)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

import lightning as L
import torch
from torch import Tensor, nn
from torchmetrics.detection import MeanAveragePrecision

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from objdet.core.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_WEIGHT_DECAY,
    ClassIndexMode,
)
from objdet.core.logging import get_logger
from objdet.core.types import DetectionPrediction, DetectionTarget

logger = get_logger(__name__)


class BaseLightningDetector(L.LightningModule):
    """Abstract base class for object detection models.

    This class provides common functionality for all detection models:
    - Standard training/validation/test step implementations
    - Metric computation (mAP)
    - Optimizer and scheduler configuration
    - Logging integration

    Subclasses must implement:
    - `forward()`: Model forward pass
    - `_build_model()`: Model architecture construction

    Args:
        num_classes: Number of object classes to detect (excluding background
            for TorchVision models, including all classes for YOLO).
        class_index_mode: How class indices are handled. TORCHVISION expects
            background at index 0, YOLO has no background class.
        learning_rate: Initial learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        confidence_threshold: Minimum confidence for predictions.
        nms_threshold: IoU threshold for NMS.
        pretrained: Whether to use pretrained weights.
        pretrained_backbone: Whether to use pretrained backbone only.

    Attributes:
        num_classes: Number of detection classes.
        class_index_mode: Class index handling mode.
        hparams: Hyperparameters (auto-saved by Lightning).

    Example:
        >>> model = MyDetector(num_classes=80, pretrained=True)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        num_classes: int,
        class_index_mode: ClassIndexMode | str = ClassIndexMode.TORCHVISION,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        optimizer: str = "adamw",
        scheduler: str | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Convert string to enum if needed
        if isinstance(class_index_mode, str):
            class_index_mode = ClassIndexMode(class_index_mode)

        self.num_classes = num_classes
        self.class_index_mode = class_index_mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.pretrained = pretrained
        self.pretrained_backbone = pretrained_backbone
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Build the model architecture
        self.model: nn.Module = self._build_model()

        # Metrics
        self._val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )
        self._test_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )

        logger.info(
            f"Initialized {self.__class__.__name__}",
            num_classes=num_classes,
            class_index_mode=class_index_mode.value,
        )

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build and return the model architecture.

        This method must be implemented by subclasses to construct
        the specific detection model architecture.

        Returns:
            The constructed PyTorch module.
        """
        ...

    @abstractmethod
    def forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget] | None = None,
    ) -> dict[str, Tensor] | list[DetectionPrediction]:
        """Forward pass of the model.

        Args:
            images: List of image tensors, each of shape (C, H, W).
            targets: Optional list of target dictionaries for training.
                Each target contains 'boxes' and 'labels' at minimum.

        Returns:
            During training (targets provided): Dictionary of losses.
            During inference (no targets): List of prediction dictionaries
                containing 'boxes', 'labels', and 'scores'.
        """
        ...

    def training_step(
        self,
        batch: tuple[list[Tensor], list[DetectionTarget]],
        batch_idx: int,
    ) -> Tensor:
        """Perform a single training step.

        Args:
            batch: Tuple of (images, targets).
            batch_idx: Index of the current batch.

        Returns:
            Total loss for backpropagation.
        """
        images, targets = batch

        # Forward pass with targets returns losses
        loss_dict = cast("dict[str, Tensor]", self.forward(images, targets))

        # Sum all losses
        total_loss: Tensor = sum(loss_dict.values())  # type: ignore[assignment]

        # Log losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, prog_bar=False, batch_size=len(images))
        self.log("train/loss", total_loss, prog_bar=True, batch_size=len(images))

        return total_loss

    def validation_step(
        self,
        batch: tuple[list[Tensor], list[DetectionTarget]],
        batch_idx: int,
    ) -> None:
        """Perform a single validation step.

        Args:
            batch: Tuple of (images, targets).
            batch_idx: Index of the current batch.
        """
        images, targets = batch

        # Get predictions (inference mode)
        predictions = cast("list[DetectionPrediction]", self.forward(images))

        # Filter predictions by confidence
        filtered_preds = self._filter_predictions(predictions)

        # Update metrics - use type ignore since types are structurally compatible
        self._val_map.update(filtered_preds, targets)  # type: ignore[arg-type]

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        metrics = self._val_map.compute()

        # Log main metrics
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True)
        self.log("val/mAP_75", metrics["map_75"], prog_bar=False)

        # Log per-size metrics if available
        if "map_small" in metrics:
            self.log("val/mAP_small", metrics["map_small"])
            self.log("val/mAP_medium", metrics["map_medium"])
            self.log("val/mAP_large", metrics["map_large"])

        # Reset metrics for next epoch
        self._val_map.reset()

    def test_step(
        self,
        batch: tuple[list[Tensor], list[DetectionTarget]],
        batch_idx: int,
    ) -> None:
        """Perform a single test step.

        Args:
            batch: Tuple of (images, targets).
            batch_idx: Index of the current batch.
        """
        images, targets = batch
        predictions = cast("list[DetectionPrediction]", self.forward(images))
        filtered_preds = self._filter_predictions(predictions)
        self._test_map.update(filtered_preds, targets)  # type: ignore[arg-type]

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at epoch end."""
        metrics = self._test_map.compute()

        self.log("test/mAP", metrics["map"])
        self.log("test/mAP_50", metrics["map_50"])
        self.log("test/mAP_75", metrics["map_75"])

        self._test_map.reset()

    def predict_step(
        self,
        batch: list[Tensor] | tuple[list[Tensor], Any],
        batch_idx: int,
    ) -> list[DetectionPrediction]:
        """Perform a single prediction step.

        Args:
            batch: List of images or tuple of (images, ...).
            batch_idx: Index of the current batch.

        Returns:
            List of prediction dictionaries.
        """
        # Handle both (images,) and (images, targets) formats
        if isinstance(batch, tuple):
            images = batch[0]
        else:
            images = batch

        predictions = cast("list[DetectionPrediction]", self.forward(images))
        return self._filter_predictions(predictions)

    def _filter_predictions(
        self,
        predictions: list[DetectionPrediction],
    ) -> list[DetectionPrediction]:
        """Filter predictions by confidence threshold.

        Args:
            predictions: List of prediction dictionaries.

        Returns:
            Filtered predictions.
        """
        filtered = []
        for pred in predictions:
            mask = pred["scores"] >= self.confidence_threshold
            filtered.append(
                {
                    "boxes": pred["boxes"][mask],
                    "labels": pred["labels"][mask],
                    "scores": pred["scores"][mask],
                }
            )
        return filtered

    def configure_optimizers(self) -> "OptimizerLRSchedulerConfig":  # type: ignore[override]
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and optional lr_scheduler configuration.
        """
        # Build optimizer
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        config: dict[str, Any] = {"optimizer": optimizer}

        # Build scheduler if specified
        if self.scheduler_name is not None:
            scheduler = self._build_scheduler(optimizer)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/mAP",
                "interval": "epoch",
                "frequency": 1,
            }

        return cast("OptimizerLRSchedulerConfig", config)

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule.

        Returns:
            Configured LR scheduler.
        """
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_kwargs.get("T_max", 100),
                eta_min=self.scheduler_kwargs.get("eta_min", 1e-6),
            )
        elif self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_kwargs.get("step_size", 30),
                gamma=self.scheduler_kwargs.get("gamma", 0.1),
            )
        elif self.scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.scheduler_kwargs.get("factor", 0.1),
                patience=self.scheduler_kwargs.get("patience", 10),
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    @property
    def num_model_classes(self) -> int:
        """Get the number of classes expected by the model.

        For TorchVision models, this includes the background class.
        For YOLO models, this is the same as num_classes.

        Returns:
            Number of classes for the model architecture.
        """
        if self.class_index_mode == ClassIndexMode.TORCHVISION:
            return self.num_classes + 1  # +1 for background
        return self.num_classes

    def get_model_info(self) -> dict[str, Any]:
        """Get model information dictionary.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_type": self.__class__.__name__,
            "num_classes": self.num_classes,
            "class_index_mode": self.class_index_mode.value,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
