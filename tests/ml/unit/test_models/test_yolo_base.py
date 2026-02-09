"""Unit tests for YOLOBaseLightning."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from objdet.core.exceptions import ModelError
from objdet.core.types import DetectionTarget
from objdet.models.yolo.base import YOLOBaseLightning


class MockYOLO(YOLOBaseLightning):
    """Concrete implementation of abstract base class for testing."""

    def _get_model_variant(self) -> str:
        return "yolov8n.pt"


@pytest.fixture
def mock_ultralytics():
    """Mock the ultralytics YOLO class."""
    with patch("objdet.models.yolo.base.YOLO") as mock:
        # Mock the underlying PyTorch model
        mock_model = MagicMock(spec=nn.Module)
        mock_model.loss = MagicMock(return_value=(torch.tensor(1.0), torch.tensor([1.0, 0.0, 0.0])))
        # Mock the head structure for modification tests
        mock_head = MagicMock()
        mock_head.nc = 80
        mock_model.model = [MagicMock(), mock_head]  # Mock modules list

        # Mock the predict method
        mock_result = MagicMock()
        mock_result.boxes.xyxy = torch.tensor([[0, 0, 10, 10]])
        mock_result.boxes.cls = torch.tensor([0])
        mock_result.boxes.conf = torch.tensor([0.9])

        mock_instance = mock.return_value
        mock_instance.model = mock_model
        mock_instance.predict.return_value = [mock_result]

        yield mock


class TestYOLOBase:
    """Tests for YOLOBaseLightning."""

    def test_init_validates_size(self):
        """Test initialization validates model size."""
        MockYOLO.MODEL_VARIANTS = {"n": "yolov8n.pt"}

        # Valid size
        model = MockYOLO(num_classes=80, model_size="n", pretrained=False)
        assert model.model_size == "n"

        # Invalid size
        with pytest.raises(ModelError):
            MockYOLO(num_classes=80, model_size="z", pretrained=False)

    def test_build_model_custom_classes(self, mock_ultralytics):
        """Test model building with custom classes modifies head."""
        MockYOLO.MODEL_VARIANTS = {"n": "yolov8n.pt"}

        # Initialize should trigger build_model
        MockYOLO(num_classes=10, model_size="n", pretrained=True)

        # Should call YOLO constructor
        mock_ultralytics.assert_called_with("yolov8n.pt")

        # Check if head modification was attempted
        # access the mock model created in fixture
        mock_instance = mock_ultralytics.return_value
        head = mock_instance.model.model[-1]
        assert head.nc == 10

    def test_forward_inference(self, mock_ultralytics):
        """Test forward pass in inference mode."""
        MockYOLO.MODEL_VARIANTS = {"n": "yolov8n.pt"}
        model = MockYOLO(num_classes=80, model_size="n")
        model.eval()

        images = [torch.rand(3, 640, 640)]
        predictions = model(images)

        assert len(predictions) == 1
        assert "boxes" in predictions[0]
        assert "labels" in predictions[0]
        assert "scores" in predictions[0]

    def test_forward_training(self, mock_ultralytics):
        """Test forward pass in training mode."""
        MockYOLO.MODEL_VARIANTS = {"n": "yolov8n.pt"}
        model = MockYOLO(num_classes=80, model_size="n")
        model.train()

        images = [torch.rand(3, 640, 640)]
        targets = [{"boxes": torch.tensor([[0, 0, 100, 100]]), "labels": torch.tensor([1])}]

        # Mock model call to return some output
        mock_ultralytics.return_value.model.return_value = torch.zeros(1, 10)

        losses = model(images, cast("list[DetectionTarget]", targets))

        assert "loss" in losses
        assert "loss_box" in losses

    def test_convert_targets(self, mock_ultralytics):
        """Test target conversion to YOLO format."""
        MockYOLO.MODEL_VARIANTS = {"n": "yolov8n.pt"}
        model = MockYOLO(num_classes=80, model_size="n")

        # Batch of 1 image, 640x640
        batch = torch.zeros(1, 3, 640, 640)
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]]), "labels": torch.tensor([5])}]

        yolo_targets = model._convert_targets_to_yolo_format(
            batch, cast("list[DetectionTarget]", targets)
        )

        # Format: [batch_idx, class, x_center, y_center, w, h]
        # x_center = 50/640, y_center = 50/640, w = 100/640, h = 100/640
        assert yolo_targets.shape == (1, 6)
        assert yolo_targets[0, 0] == 0  # batch_idx
        assert yolo_targets[0, 1] == 5  # class
