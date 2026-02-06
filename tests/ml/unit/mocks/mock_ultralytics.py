"""Mock Ultralytics YOLO module for unit testing.

This module provides mock implementations of Ultralytics YOLO classes
to enable unit testing without the heavy ultralytics dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass
class MockBoxes:
    """Mock Ultralytics Boxes class."""

    xyxy: Tensor = field(default_factory=lambda: torch.empty(0, 4))
    conf: Tensor = field(default_factory=lambda: torch.empty(0))
    cls: Tensor = field(default_factory=lambda: torch.empty(0))
    data: Tensor = field(default_factory=lambda: torch.empty(0, 6))

    def __len__(self) -> int:
        return len(self.xyxy)


@dataclass
class MockResults:
    """Mock Ultralytics Results class."""

    boxes: MockBoxes = field(default_factory=MockBoxes)
    orig_shape: tuple[int, int] = (640, 640)
    path: str = "mock_image.jpg"

    @classmethod
    def with_detections(
        cls,
        boxes: Tensor | None = None,
        scores: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MockResults:
        """Create mock results with specified detections."""
        if boxes is None:
            boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        if scores is None:
            scores = torch.tensor([0.9])
        if labels is None:
            labels = torch.tensor([0.0])

        mock_boxes = MockBoxes(
            xyxy=boxes,
            conf=scores,
            cls=labels,
            data=torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1),
        )
        return cls(boxes=mock_boxes)


class MockModel:
    """Mock internal model structure."""

    def __init__(self, num_classes: int = 80) -> None:
        self.nc = num_classes
        self.names = {i: f"class_{i}" for i in range(num_classes)}

    def __call__(self, x: Tensor) -> Tensor:
        """Mock forward pass."""
        # Return mock loss tensor for training
        return torch.tensor(0.5).requires_grad_(True)


class MockYOLO:
    """Mock Ultralytics YOLO model for testing.

    This class mimics the essential interface of ultralytics.YOLO
    without requiring the actual package installation.

    Example:
        >>> from tests.unit.mocks import MockYOLO
        >>> model = MockYOLO("yolov8n.pt")
        >>> results = model.predict("image.jpg")
    """

    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        task: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize mock YOLO model.

        Args:
            model: Model name or path (ignored in mock).
            task: Task type (ignored in mock).
            verbose: Verbosity flag (ignored in mock).
        """
        self.model_path = str(model)
        self.task = task or "detect"
        self.model = MockModel()
        self.names = self.model.names
        self.overrides: dict[str, Any] = {}
        self._trainer = None

    def __call__(self, source: Any, **kwargs: Any) -> list[MockResults]:
        """Run inference (alias for predict)."""
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        **kwargs: Any,
    ) -> list[MockResults]:
        """Mock prediction that returns sample detections.

        Args:
            source: Image source (ignored in mock).
            conf: Confidence threshold.
            iou: IoU threshold.
            imgsz: Image size.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of mock results with sample detections.
        """
        # Return one result with sample detections
        return [
            MockResults.with_detections(
                boxes=torch.tensor(
                    [
                        [100.0, 100.0, 200.0, 200.0],
                        [300.0, 300.0, 400.0, 400.0],
                    ]
                ),
                scores=torch.tensor([0.95, 0.87]),
                labels=torch.tensor([0.0, 1.0]),
            )
        ]

    def train(
        self,
        data: str | None = None,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Mock training that returns sample metrics.

        Args:
            data: Dataset config path (ignored).
            epochs: Number of epochs.
            imgsz: Image size.
            batch: Batch size.
            **kwargs: Additional arguments (ignored).

        Returns:
            Dictionary with mock training metrics.
        """
        return {
            "mAP50": 0.65,
            "mAP50-95": 0.45,
            "precision": 0.72,
            "recall": 0.68,
        }

    def val(self, data: str | None = None, **kwargs: Any) -> dict[str, float]:
        """Mock validation that returns sample metrics.

        Args:
            data: Dataset config path (ignored).
            **kwargs: Additional arguments (ignored).

        Returns:
            Dictionary with mock validation metrics.
        """
        return {
            "mAP50": 0.60,
            "mAP50-95": 0.42,
        }

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        **kwargs: Any,
    ) -> str:
        """Mock export that returns output path.

        Args:
            format: Export format.
            imgsz: Image size.
            **kwargs: Additional arguments (ignored).

        Returns:
            Mock output path string.
        """
        base = Path(self.model_path).stem
        return f"{base}.{format}"

    def info(self, verbose: bool = True) -> dict[str, Any]:
        """Return mock model info.

        Args:
            verbose: Whether to print info.

        Returns:
            Dictionary with mock model information.
        """
        return {
            "layers": 225,
            "parameters": 3_200_000,
            "gradients": 3_200_000,
            "gflops": 8.2,
        }

    def fuse(self) -> MockYOLO:
        """Mock model fusion (returns self)."""
        return self

    def to(self, device: str | torch.device) -> MockYOLO:
        """Mock device transfer (returns self)."""
        return self

    def eval(self) -> MockYOLO:
        """Set model to eval mode (returns self)."""
        return self

    def train_mode(self, mode: bool = True) -> MockYOLO:
        """Set model training mode (returns self)."""
        return self
