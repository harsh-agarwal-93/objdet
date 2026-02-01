"""Unit tests for YOLO models using mocks."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Add tests directory to path for mock imports
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from unit.mocks.mock_ultralytics import MockResults, MockYOLO  # noqa: E402, type: ignore


class TestMockYOLO:
    """Tests for the MockYOLO class itself."""

    def test_mock_yolo_init(self) -> None:
        """Test MockYOLO initialization."""
        yolo = MockYOLO("yolov8n.pt")

        assert yolo.model_path == "yolov8n.pt"
        assert yolo.task == "detect"

    def test_mock_yolo_predict(self) -> None:
        """Test MockYOLO prediction returns results."""
        yolo = MockYOLO("yolov8n.pt")

        results = yolo.predict("image.jpg")

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].boxes is not None

    def test_mock_yolo_train(self) -> None:
        """Test MockYOLO training returns metrics."""
        yolo = MockYOLO("yolov8n.pt")

        metrics = yolo.train(epochs=10)

        assert isinstance(metrics, dict)
        assert "mAP50" in metrics

    def test_mock_yolo_export(self) -> None:
        """Test MockYOLO export returns path."""
        yolo = MockYOLO("yolov8n.pt")

        output = yolo.export(format="onnx")

        assert output == "yolov8n.onnx"

    def test_mock_yolo_info(self) -> None:
        """Test MockYOLO info returns dict."""
        yolo = MockYOLO("yolov8n.pt")

        info = yolo.info()

        assert isinstance(info, dict)
        assert "layers" in info
        assert "parameters" in info


class TestMockResults:
    """Tests for MockResults class."""

    def test_mock_results_empty(self) -> None:
        """Test MockResults with no detections."""
        results = MockResults()

        assert len(results.boxes) == 0

    def test_mock_results_with_detections(self) -> None:
        """Test MockResults with sample detections."""
        boxes = torch.tensor([[10.0, 20.0, 100.0, 150.0]])
        scores = torch.tensor([0.95])
        labels = torch.tensor([1.0])

        results = MockResults.with_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
        )

        assert len(results.boxes) == 1
        assert torch.equal(results.boxes.xyxy, boxes)
        assert torch.equal(results.boxes.conf, scores)


class TestYOLOv8WithMock:
    """Tests for YOLOv8 using mocked Ultralytics."""

    @pytest.fixture
    def mock_ultralytics(self):
        """Set up mocked Ultralytics YOLO."""
        with patch("ultralytics.YOLO", MockYOLO):
            yield

    def test_yolov8_model_sizes(self) -> None:
        """Test YOLOv8 model size variants."""
        from objdet.models.yolo.yolov8 import YOLOv8

        # Check MODEL_VARIANTS is defined
        assert hasattr(YOLOv8, "MODEL_VARIANTS")
        assert len(YOLOv8.MODEL_VARIANTS) > 0
        assert "n" in YOLOv8.MODEL_VARIANTS
        assert "s" in YOLOv8.MODEL_VARIANTS

    def test_yolov11_model_sizes(self) -> None:
        """Test YOLOv11 model size variants."""
        from objdet.models.yolo.yolov11 import YOLOv11

        # Check MODEL_VARIANTS is defined
        assert hasattr(YOLOv11, "MODEL_VARIANTS")
        assert len(YOLOv11.MODEL_VARIANTS) > 0


class TestYOLOIntegrationWithMock:
    """Integration tests for YOLO with mocked backend."""

    def test_mock_predict_returns_boxes(self) -> None:
        """Test that mock prediction returns valid boxes."""
        yolo = MockYOLO()

        results = yolo.predict("test.jpg")
        boxes = results[0].boxes

        # Should have sample detections
        assert len(boxes.xyxy) > 0
        assert boxes.xyxy.shape[1] == 4  # x1, y1, x2, y2

    def test_mock_predict_returns_scores(self) -> None:
        """Test that mock prediction returns confidence scores."""
        yolo = MockYOLO()

        results = yolo.predict("test.jpg")
        boxes = results[0].boxes

        assert len(boxes.conf) > 0
        assert all(0 <= score <= 1 for score in boxes.conf)

    def test_mock_predict_returns_labels(self) -> None:
        """Test that mock prediction returns class labels."""
        yolo = MockYOLO()

        results = yolo.predict("test.jpg")
        boxes = results[0].boxes

        assert len(boxes.cls) > 0
