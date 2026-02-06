"""Unit tests for RetinaNet model."""

from __future__ import annotations

import pytest
import torch


class TestRetinaNetInit:
    """Tests for RetinaNet initialization."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        from objdet.models.torchvision.retinanet import RetinaNet

        model = RetinaNet(
            num_classes=10,
            pretrained=False,
            pretrained_backbone=False,
        )

        assert model.num_classes == 10

    def test_init_with_pretrained_backbone(self) -> None:
        """Test initialization with pretrained backbone."""
        from objdet.models.torchvision.retinanet import RetinaNet

        model = RetinaNet(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=True,
        )

        assert model is not None


class TestRetinaNetForward:
    """Tests for RetinaNet forward pass."""

    @pytest.fixture
    def retinanet_model(self):
        """Create a RetinaNet model for testing."""
        from objdet.models.torchvision.retinanet import RetinaNet

        model = RetinaNet(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=False,
        )
        return model

    def test_forward_training_returns_losses(self, retinanet_model) -> None:
        """Test forward returns losses in training mode."""
        retinanet_model.train()

        images = [torch.rand(3, 320, 320)]
        targets = [
            {
                "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        losses = retinanet_model(images, targets)

        assert isinstance(losses, dict)
        assert len(losses) > 0

    def test_forward_inference_returns_predictions(self, retinanet_model) -> None:
        """Test forward returns predictions in eval mode."""
        retinanet_model.eval()

        images = [torch.rand(3, 320, 320)]

        with torch.no_grad():
            predictions = retinanet_model(images)

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert "boxes" in predictions[0]
        assert "labels" in predictions[0]
        assert "scores" in predictions[0]

    def test_forward_batch(self, retinanet_model) -> None:
        """Test forward with batch of images."""
        retinanet_model.eval()

        batch_size = 3
        images = [torch.rand(3, 320, 320) for _ in range(batch_size)]

        with torch.no_grad():
            predictions = retinanet_model(images)

        assert len(predictions) == batch_size


class TestRetinaNetProperties:
    """Tests for RetinaNet properties."""

    def test_num_classes_property(self) -> None:
        """Test num_classes property."""
        from objdet.models.torchvision.retinanet import RetinaNet

        model = RetinaNet(
            num_classes=15,
            pretrained=False,
            pretrained_backbone=False,
        )

        assert model.num_classes == 15

    def test_mode_switching(self) -> None:
        """Test switching between train and eval modes."""
        from objdet.models.torchvision.retinanet import RetinaNet

        model = RetinaNet(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=False,
        )

        model.train()
        assert model.training

        model.eval()
        assert not model.training
