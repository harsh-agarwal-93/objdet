from typing import cast

import pytest
import torch
from torch import nn

from objdet.core.types import DetectionPrediction
from objdet.models.ensemble.nms import NMSEnsemble


class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x


@pytest.fixture
def mock_models() -> list[DummyModel]:
    """Create a list of dummy models with consistent num_classes."""
    return [DummyModel(3), DummyModel(3)]


@pytest.fixture
def sample_predictions() -> list[DetectionPrediction]:
    """Create sample predictions for two models."""
    return [
        cast(
            "DetectionPrediction",
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]]),
                "labels": torch.tensor([1, 2]),
                "scores": torch.tensor([0.9, 0.8]),
            },
        ),
        cast(
            "DetectionPrediction",
            {
                "boxes": torch.tensor([[12.0, 12.0, 52.0, 52.0], [150.0, 150.0, 200.0, 200.0]]),
                "labels": torch.tensor([1, 3]),
                "scores": torch.tensor([0.85, 0.7]),
            },
        ),
    ]


def test_nms_ensemble_init(mock_models: list[DummyModel]) -> None:
    """Test NMSEnsemble initialization."""
    ensemble = NMSEnsemble(models=mock_models, iou_thresh=0.6)
    assert ensemble.iou_thresh == 0.6
    assert not ensemble.use_soft_nms
    assert len(ensemble.ensemble_models) == 2


def test_soft_nms_ensemble_init(mock_models: list[DummyModel]) -> None:
    """Test NMSEnsemble initialization with Soft-NMS."""
    ensemble = NMSEnsemble(
        models=mock_models,
        soft_nms=True,
        soft_nms_sigma=0.6,
        soft_nms_method="linear",
    )
    assert ensemble.use_soft_nms
    assert ensemble.soft_nms_sigma == 0.6
    assert ensemble.soft_nms_method == "linear"


def test_fuse_predictions_nms(
    mock_models: list[DummyModel], sample_predictions: list[DetectionPrediction]
) -> None:
    """Test _fuse_predictions with standard NMS."""
    ensemble = NMSEnsemble(models=mock_models, iou_thresh=0.5)
    image_size = (1000, 1000)

    # We need to mock the external ensemble_boxes.nms call because we want to test
    # the NMSEnsemble class logic (normalization, denormalization, etc.)
    # However, for coverage we want to run the actual function.
    # Since we are in a unit test, we should ideally use the real one if dependencies are there.
    # The previous tests used mock_ensemble_boxes, but here we want ACTUAL coverage.

    result = ensemble._fuse_predictions(sample_predictions, image_size)

    assert "boxes" in result
    assert "labels" in result
    assert "scores" in result

    assert isinstance(result["boxes"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)
    assert isinstance(result["scores"], torch.Tensor)

    # Check that boxes were denormalized back (should be in range beyond [0, 1])
    if result["boxes"].numel() > 0:
        assert torch.any(result["boxes"] > 1.0)


def test_fuse_predictions_soft_nms(
    mock_models: list[DummyModel], sample_predictions: list[DetectionPrediction]
) -> None:
    """Test _fuse_predictions with Soft-NMS."""
    ensemble = NMSEnsemble(models=mock_models, soft_nms=True)
    image_size = (1000, 1000)

    result = ensemble._fuse_predictions(sample_predictions, image_size)

    assert isinstance(result["boxes"], torch.Tensor)
    assert result["boxes"].shape[1] == 4


def test_fuse_predictions_empty(mock_models: list[DummyModel]) -> None:
    """Test _fuse_predictions with empty input."""
    ensemble = NMSEnsemble(models=mock_models)
    image_size = (1000, 1000)

    empty_preds = [
        cast(
            "DetectionPrediction",
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            },
        ),
        cast(
            "DetectionPrediction",
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            },
        ),
    ]

    result = ensemble._fuse_predictions(empty_preds, image_size)

    assert result["boxes"].numel() == 0
    assert result["labels"].numel() == 0
    assert result["scores"].numel() == 0


def test_fuse_predictions_clamping(mock_models: list[DummyModel]) -> None:
    """Test that boxes are clamped to [0, 1] before fusion."""
    ensemble = NMSEnsemble(models=mock_models)
    image_size = (100, 100)

    # Boxes outside [0, 100]
    preds = [
        cast(
            "DetectionPrediction",
            {
                "boxes": torch.tensor([[-10.0, -10.0, 110.0, 110.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            },
        )
    ]

    result = ensemble._fuse_predictions(preds, image_size)

    # Clamped to 0-100
    assert torch.all(result["boxes"] >= 0)
    assert torch.all(result["boxes"] <= 100)
