"""Detection visualization callback.

This callback visualizes model predictions on sample images during
training, useful for monitoring model progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from objdet.core.logging import get_logger

logger = get_logger(__name__)


class DetectionVisualizationCallback(Callback):
    """Callback to visualize detection predictions on sample images.

    Args:
        num_samples: Number of samples to visualize per epoch.
        save_dir: Directory to save visualization images.
        log_to_tensorboard: Whether to also log to TensorBoard.
        confidence_threshold: Minimum confidence for visualization.
        class_names: Optional list of class names for labels.
        box_color: Color for prediction boxes (BGR tuple or "random").

    Example:
        >>> callback = DetectionVisualizationCallback(
        ...     num_samples=8,
        ...     save_dir="outputs/visualizations",
        ...     class_names=["person", "car", "dog"],
        ... )
    """

    def __init__(
        self,
        num_samples: int = 8,
        save_dir: str | Path = "outputs/visualizations",
        log_to_tensorboard: bool = True,
        confidence_threshold: float = 0.5,
        class_names: list[str] | None = None,
        box_color: tuple[int, int, int] | str = (0, 255, 0),
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.log_to_tensorboard = log_to_tensorboard
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self.box_color = box_color

        self._validation_samples: list[tuple[Tensor, dict]] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect samples for visualization."""
        if len(self._validation_samples) >= self.num_samples:
            return

        images, targets = batch

        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module(images)

        # Store samples
        for i in range(min(len(images), self.num_samples - len(self._validation_samples))):
            self._validation_samples.append(
                (
                    images[i].cpu(),
                    predictions[i],
                )
            )

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Create and save visualizations."""
        if not self._validation_samples:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        visualizations = []

        for idx, (image, prediction) in enumerate(self._validation_samples):
            # Create visualization
            vis_image = self._draw_predictions(image, prediction)
            visualizations.append(vis_image)

            # Save to file
            save_path = self.save_dir / f"epoch_{trainer.current_epoch}_sample_{idx}.png"
            self._save_image(vis_image, save_path)

        # Log to TensorBoard if available
        if self.log_to_tensorboard and trainer.logger:
            self._log_to_tensorboard(trainer, visualizations)

        logger.info(f"Saved {len(visualizations)} visualization images to {self.save_dir}")

        # Clear samples for next epoch
        self._validation_samples.clear()

    def _draw_predictions(
        self,
        image: Tensor,
        prediction: dict[str, Tensor],
    ) -> Tensor:
        """Draw bounding boxes on image.

        Args:
            image: Image tensor (C, H, W) normalized [0, 1].
            prediction: Prediction dict with boxes, labels, scores.

        Returns:
            Image tensor with drawn boxes.
        """
        # Convert to 0-255 range for drawing
        image = (image * 255).byte()

        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]

        # Filter by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Draw boxes using torchvision
        try:
            from torchvision.utils import draw_bounding_boxes

            # Create labels
            if self.class_names:
                box_labels = [
                    f"{self.class_names[lbl.item()]}: {s:.2f}"
                    for lbl, s in zip(labels, scores, strict=True)
                ]
            else:
                box_labels = [
                    f"class_{lbl.item()}: {s:.2f}" for lbl, s in zip(labels, scores, strict=True)
                ]

            image = draw_bounding_boxes(
                image,
                boxes,
                labels=box_labels,
                colors="green",
                width=2,
            )

        except ImportError:
            # Fallback: just return image without boxes
            logger.warning("torchvision.utils not available for drawing")

        return image

    def _save_image(self, image: Tensor, path: Path) -> None:
        """Save image tensor to file."""
        try:
            from torchvision.io import write_png

            write_png(image, str(path))
        except ImportError:
            # Fallback using PIL
            from PIL import Image as PILImage

            img_np = image.permute(1, 2, 0).numpy()
            PILImage.fromarray(img_np).save(path)

    def _log_to_tensorboard(
        self,
        trainer: L.Trainer,
        images: list[Tensor],
    ) -> None:
        """Log images to TensorBoard."""
        try:
            # Stack images into grid
            from torchvision.utils import make_grid

            grid = make_grid(
                [img.float() / 255.0 for img in images],
                nrow=4,
                padding=2,
            )

            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.add_image(
                    "val/predictions",
                    grid,
                    trainer.current_epoch,
                )
        except Exception as e:
            logger.debug(f"Could not log to TensorBoard: {e}")
