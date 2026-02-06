"""Unit tests for FasterRCNN model."""

from __future__ import annotations

import pytest
import torch


class TestFasterRCNNInit:
    """Tests for FasterRCNN initialization."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        from objdet.models.torchvision.faster_rcnn import FasterRCNN

        model = FasterRCNN(
            num_classes=10,
            pretrained=False,
            pretrained_backbone=False,
        )

        assert model.num_classes == 10

    def test_init_with_pretrained_backbone(self) -> None:
        """Test initialization with pretrained backbone."""
        from objdet.models.torchvision.faster_rcnn import FasterRCNN

        model = FasterRCNN(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=True,
        )

        assert model is not None


class TestFasterRCNNForward:
    """Tests for FasterRCNN forward pass."""

    @pytest.fixture
    def frcnn_model(self):
        """Create a FasterRCNN model for testing."""
        from objdet.models.torchvision.faster_rcnn import FasterRCNN

        model = FasterRCNN(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=False,
        )
        return model

    def test_forward_training_returns_losses(self, frcnn_model) -> None:
        """Test forward returns losses in training mode."""
        frcnn_model.train()

        images = [torch.rand(3, 320, 320)]
        targets = [
            {
                "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        losses = frcnn_model(images, targets)

        assert isinstance(losses, dict)
        # Should have classification and regression losses
        assert len(losses) > 0

    def test_forward_inference_returns_predictions(self, frcnn_model) -> None:
        """Test forward returns predictions in eval mode."""
        frcnn_model.eval()

        images = [torch.rand(3, 320, 320)]

        with torch.no_grad():
            predictions = frcnn_model(images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert "boxes" in predictions[0]
        assert "labels" in predictions[0]
        assert "scores" in predictions[0]

    def test_forward_batch(self, frcnn_model) -> None:
        """Test forward with batch of images."""
        frcnn_model.eval()

        batch_size = 4
        images = [torch.rand(3, 320, 320) for _ in range(batch_size)]

        with torch.no_grad():
            predictions = frcnn_model(images)

        assert len(predictions) == batch_size


class TestFasterRCNNProperties:
    """Tests for FasterRCNN properties and utilities."""

    def test_num_classes_property(self) -> None:
        """Test num_classes property."""
        from objdet.models.torchvision.faster_rcnn import FasterRCNN

        model = FasterRCNN(
            num_classes=20,
            pretrained=False,
            pretrained_backbone=False,
        )

        assert model.num_classes == 20

    def test_eval_mode(self) -> None:
        """Test switching to eval mode."""
        from objdet.models.torchvision.faster_rcnn import FasterRCNN

        model = FasterRCNN(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=False,
        )

        model.eval()
        assert not model.training

    def test_train_mode(self) -> None:
        """Test switching to train mode."""
        from objdet.models.torchvision.faster_rcnn import FasterRCNN

        model = FasterRCNN(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=False,
        )

        model.eval()
        model.train()
        assert model.training
