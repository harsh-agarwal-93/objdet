"""High-level predictor for object detection inference.

This module provides a simple API for running inference with
trained detection models.

Example:
    >>> from objdet.inference import Predictor
    >>>
    >>> predictor = Predictor.from_checkpoint("model.ckpt")
    >>> results = predictor.predict("image.jpg")
    >>> predictor.predict_batch(["img1.jpg", "img2.jpg"])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from objdet.core.exceptions import InferenceError, ModelError
from objdet.core.logging import get_logger

if TYPE_CHECKING:
    from PIL import Image

    from objdet.core.types import DetectionPrediction
    from objdet.models.base import BaseLightningDetector

logger = get_logger(__name__)


class Predictor:
    """High-level inference predictor for detection models.

    Provides a simple interface for loading models and running inference
    on images or directories.

    Args:
        model: Detection model instance.
        device: Device to run inference on.
        confidence_threshold: Minimum confidence for predictions.
        nms_threshold: IoU threshold for NMS.

    Example:
        >>> predictor = Predictor.from_checkpoint("model.ckpt")
        >>> result = predictor.predict("photo.jpg")
        >>> print(f"Found {len(result['boxes'])} objects")
    """

    def __init__(
        self,
        model: BaseLightningDetector,
        device: str | torch.device = "cuda",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Predictor initialized: model={model.__class__.__name__}, device={device}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> Predictor:
        """Create predictor from a Lightning checkpoint.

        Args:
            checkpoint_path: Path to .ckpt checkpoint file.
            model_class: Optional model class. If None, tries to infer from checkpoint.
            device: Device for inference.
            **kwargs: Additional arguments for Predictor.

        Returns:
            Configured Predictor instance.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ModelError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Determine model class
        if model_class is None:
            # Try to get from checkpoint
            if "hyper_parameters" in checkpoint:
                hparams = checkpoint["hyper_parameters"]
                model_type = hparams.get("model_type")

                # Import model from registry
                from objdet.models import MODEL_REGISTRY

                if model_type and model_type.lower() in MODEL_REGISTRY:
                    model_class = MODEL_REGISTRY.get(model_type.lower())
                else:
                    raise ModelError(
                        "Cannot determine model class from checkpoint. "
                        "Please provide model_class explicitly."
                    )
            else:
                raise ModelError("No hyper_parameters in checkpoint")

        # Load model
        model = model_class.load_from_checkpoint(checkpoint_path)

        return cls(model=model, device=device, **kwargs)

    def predict(
        self,
        image: str | Path | Tensor | Image.Image,
        return_image: bool = False,
    ) -> DetectionPrediction | tuple[DetectionPrediction, Tensor]:
        """Run inference on a single image.

        Args:
            image: Image path, PIL Image, or tensor.
            return_image: Whether to also return the preprocessed image.

        Returns:
            Prediction dict with boxes, labels, scores.
            If return_image, returns tuple of (prediction, image_tensor).
        """
        # Load and preprocess image
        image_tensor = self._load_image(image)
        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])

        # Post-process
        result = self._postprocess(predictions[0])

        if return_image:
            return result, image_tensor
        return result

    def predict_batch(
        self,
        images: list[str | Path | Tensor],
        batch_size: int = 8,
    ) -> list[DetectionPrediction]:
        """Run inference on multiple images.

        Args:
            images: List of image paths or tensors.
            batch_size: Batch size for inference.

        Returns:
            List of prediction dictionaries.
        """
        all_results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Load and stack images
            tensors = [self._load_image(img).to(self.device) for img in batch_images]

            # Run inference
            with torch.no_grad():
                predictions = self.model(tensors)

            # Post-process each
            for pred in predictions:
                all_results.append(self._postprocess(pred))

        return all_results

    def predict_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        batch_size: int = 8,
    ) -> dict[str, DetectionPrediction]:
        """Run inference on all images in a directory.

        Args:
            directory: Path to directory.
            extensions: Image file extensions to include.
            batch_size: Batch size for inference.

        Returns:
            Dict mapping filename to prediction.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise InferenceError(f"Not a directory: {directory}")

        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

        image_paths = sorted(set(image_paths))
        logger.info(f"Found {len(image_paths)} images in {directory}")

        # Run batch inference
        from typing import cast

        predictions = self.predict_batch(
            cast("list[str | Path | Tensor]", image_paths), batch_size=batch_size
        )

        # Map to filenames
        return {path.name: pred for path, pred in zip(image_paths, predictions, strict=True)}

    def _load_image(self, image: str | Path | Tensor | Image.Image) -> Tensor:
        """Load and preprocess image to tensor."""
        if isinstance(image, Tensor):
            return image

        if isinstance(image, (str, Path)):
            from PIL import Image as PILImage

            image = PILImage.open(image).convert("RGB")

        # Convert PIL to tensor
        import numpy as np

        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor

    def _postprocess(self, prediction: DetectionPrediction) -> DetectionPrediction:
        """Post-process prediction (filter by confidence)."""
        scores = prediction["scores"]
        mask = scores >= self.confidence_threshold

        return {
            "boxes": prediction["boxes"][mask].cpu(),
            "labels": prediction["labels"][mask].cpu(),
            "scores": scores[mask].cpu(),
        }
