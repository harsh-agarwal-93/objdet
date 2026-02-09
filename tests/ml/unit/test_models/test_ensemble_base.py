"""Unit tests for BaseEnsemble."""

import pytest
import torch
from torch import Tensor, nn

from objdet.core.types import DetectionPrediction, DetectionTarget
from objdet.models.base import BaseLightningDetector
from objdet.models.ensemble.base import BaseEnsemble


class MockDetector(BaseLightningDetector):
    """Mock detector for ensemble testing."""

    def __init__(self, num_classes: int = 80):
        super().__init__(num_classes=num_classes)
        self.dummy_param = nn.Parameter(torch.tensor([1.0]))

    def _build_model(self) -> nn.Module:
        return nn.Identity()

    def forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget] | None = None,
    ) -> dict[str, Tensor] | list[DetectionPrediction]:
        if self.training and targets is not None:
            return {"loss": torch.tensor(1.0)}
        else:
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
                for _ in images
            ]


class ConcreteEnsemble(BaseEnsemble):
    """Concrete ensemble for testing abstract base."""

    def _fuse_predictions(self, predictions, image_size):
        # specific implementation is tested in WBF/NMS tests
        # here we just return the first prediction
        return predictions[0]


class TestBaseEnsemble:
    """Tests for BaseEnsemble."""

    def test_init_validation(self):
        """Test initialization consistency checks."""
        m1 = MockDetector(num_classes=10)
        m2 = MockDetector(num_classes=10)
        m3 = MockDetector(num_classes=20)

        # Valid
        ConcreteEnsemble(models=[m1, m2])

        # Mismatch classes
        with pytest.raises(ValueError):
            ConcreteEnsemble(models=[m1, m3])

        # Mismatch weights
        with pytest.raises(ValueError):
            ConcreteEnsemble(models=[m1, m2], weights=[0.5])

    def test_weights_normalization(self):
        """Test weights are normalized."""
        m1 = MockDetector(num_classes=10)
        m2 = MockDetector(num_classes=10)

        ensemble = ConcreteEnsemble(models=[m1, m2], weights=[2.0, 8.0])
        assert ensemble.weights == [0.2, 0.8]

    def test_training_forward(self):
        """Test training forward aggregates losses."""
        m1 = MockDetector(num_classes=10)
        m2 = MockDetector(num_classes=10)
        ensemble = ConcreteEnsemble(models=[m1, m2])
        ensemble.train()

        images = [torch.rand(3, 100, 100)]
        targets = [{"boxes": [], "labels": []}]

        losses = ensemble(images, targets)

        assert "loss" in losses
        # Default equal weights: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert losses["loss"].item() == 1.0
        assert "model_0/loss" in losses

    def test_inference_forward(self):
        """Test inference forward fuses predictions."""
        m1 = MockDetector(num_classes=10)
        m2 = MockDetector(num_classes=10)
        ensemble = ConcreteEnsemble(models=[m1, m2])
        ensemble.eval()

        images = [torch.rand(3, 100, 100)]
        preds = ensemble(images)

        assert len(preds) == 1
        assert "boxes" in preds[0]
