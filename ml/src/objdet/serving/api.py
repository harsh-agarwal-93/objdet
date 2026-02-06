"""Detection API implementation for LitServe.

This module provides the LitServe API class for serving
object detection models via REST endpoints.

Example:
    >>> from objdet.serving.api import DetectionAPI
    >>>
    >>> api = DetectionAPI(checkpoint="model.ckpt")
    >>> server = ls.LitServer(api, accelerator="cuda")
    >>> server.run(port=8000)
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from objdet.core.logging import get_logger

logger = get_logger(__name__)


class DetectionAPI:
    """LitServe API for object detection.

    This class implements the LitServe API interface for serving
    detection models. Supports dynamic batching and async processing.

    Args:
        checkpoint_path: Path to model checkpoint.
        model_class: Model class (if not inferrable from checkpoint).
        device: Device for inference.
        confidence_threshold: Minimum confidence for predictions.
        max_batch_size: Maximum batch size for dynamic batching.

    Example:
        >>> api = DetectionAPI(checkpoint="model.ckpt")
        >>> # Use with LitServer
        >>> import litserve as ls
        >>> server = ls.LitServer(api, accelerator="cuda")
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_class: type | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        max_batch_size: int = 8,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_class = model_class
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_batch_size = max_batch_size

        # Model loaded in setup()
        self.model = None
        self.class_names: list[str] = []

    def setup(self, device: str) -> None:
        """Setup method called by LitServe.

        Args:
            device: Device assigned by LitServe.
        """
        from objdet.inference.predictor import Predictor

        self.predictor = Predictor.from_checkpoint(
            self.checkpoint_path,
            model_class=self.model_class,
            device=device,
            confidence_threshold=self.confidence_threshold,
        )

        self.model = self.predictor.model

        # Get class names if available
        if hasattr(self.model, "hparams") and "class_names" in self.model.hparams:
            self.class_names = self.model.hparams["class_names"]

        logger.info(f"Detection API loaded model on {device}")

    def decode_request(self, request: dict[str, Any]) -> Tensor:
        """Decode incoming request to image tensor.

        Supports:
        - Base64-encoded images in 'image' field
        - Image URLs in 'url' field
        - Raw tensor data in 'tensor' field

        Args:
            request: Request dictionary.

        Returns:
            Image tensor (C, H, W).
        """
        if "image" in request:
            # Base64 encoded image
            import numpy as np
            from PIL import Image

            image_data = base64.b64decode(request["image"])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor

        elif "url" in request:
            # URL - fetch image
            import httpx
            import numpy as np
            from PIL import Image

            response = httpx.get(request["url"], timeout=10)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor

        elif "tensor" in request:
            # Raw tensor data
            return torch.tensor(request["tensor"])

        else:
            raise ValueError("Request must contain 'image', 'url', or 'tensor' field")

    def predict(self, inputs: list[Tensor]) -> list[dict[str, Tensor]]:
        """Run prediction on batch of inputs.

        Args:
            inputs: List of image tensors.

        Returns:
            List of prediction dictionaries.
        """
        # Run batch prediction - type ignores for structural typing compatibility
        predictions = self.predictor.predict_batch(inputs)  # type: ignore[arg-type]
        return predictions  # type: ignore[return-value]

    def encode_response(self, output: dict[str, Tensor]) -> dict[str, Any]:
        """Encode prediction to JSON-serializable response.

        Args:
            output: Prediction dictionary with tensors.

        Returns:
            JSON-serializable response.
        """
        boxes = output["boxes"].tolist()
        labels = output["labels"].tolist()
        scores = output["scores"].tolist()

        # Add class names if available
        if self.class_names:
            class_labels = [
                self.class_names[label] if label < len(self.class_names) else f"class_{label}"
                for label in labels
            ]
        else:
            class_labels = [f"class_{label}" for label in labels]

        detections = []
        for box, label, score, class_name in zip(boxes, labels, scores, class_labels, strict=True):
            detections.append(
                {
                    "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                    "class_id": label,
                    "class_name": class_name,
                    "confidence": score,
                }
            )

        return {
            "detections": detections,
            "count": len(detections),
        }


class ABTestingAPI:
    """A/B testing wrapper for multiple detection models.

    Routes requests to different model versions based on
    configured traffic splits.

    Args:
        models: Dict mapping model name to (checkpoint_path, weight).
        device: Device for inference.

    Example:
        >>> api = ABTestingAPI(
        ...     {
        ...         "v1": ("model_v1.ckpt", 0.7),  # 70% traffic
        ...         "v2": ("model_v2.ckpt", 0.3),  # 30% traffic
        ...     }
        ... )
    """

    def __init__(
        self,
        models: dict[str, tuple[str | Path, float]],
        device: str = "cuda",
    ) -> None:
        self.model_configs = models
        self.device = device
        self.apis: dict[str, DetectionAPI] = {}

        # Normalize weights
        total_weight = sum(w for _, w in models.values())
        self.weights = {name: w / total_weight for name, (_, w) in models.items()}

    def setup(self, device: str) -> None:
        """Setup all model APIs."""
        for name, (checkpoint, _) in self.model_configs.items():
            api = DetectionAPI(checkpoint_path=checkpoint, device=device)
            api.setup(device)
            self.apis[name] = api

        logger.info(f"A/B testing API loaded {len(self.apis)} models")

    def _select_model(self) -> tuple[str, DetectionAPI]:
        """Select model based on traffic weights."""
        import random

        r = random.random()
        cumulative = 0.0

        for name, weight in self.weights.items():
            cumulative += weight
            if r <= cumulative:
                return name, self.apis[name]

        # Fallback to first model
        name = list(self.apis.keys())[0]
        return name, self.apis[name]

    def decode_request(self, request: dict[str, Any]) -> Tensor:
        """Decode request using first API."""
        first_api = list(self.apis.values())[0]
        return first_api.decode_request(request)

    def predict(self, inputs: list[Tensor]) -> list[tuple[str, dict[str, Any]]]:
        """Run prediction with selected model."""
        results = []
        for inp in inputs:
            model_name, api = self._select_model()
            pred = api.predict([inp])[0]
            results.append((model_name, pred))
        return results

    def encode_response(
        self,
        output: tuple[str, dict[str, Tensor]],
    ) -> dict[str, Any]:
        """Encode response with model version info."""
        model_name, pred = output
        first_api = list(self.apis.values())[0]
        response = first_api.encode_response(pred)
        response["model_version"] = model_name
        return response
