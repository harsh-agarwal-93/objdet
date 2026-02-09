from typing import cast

import pytest
import torch
from torch import nn

from objdet.core.types import DetectionPrediction
from objdet.models.ensemble.wbf import WBFEnsemble


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


def test_wbf_ensemble_init(mock_models: list[DummyModel]) -> None:
    """Test WBFEnsemble initialization."""
    ensemble = WBFEnsemble(models=mock_models, iou_thresh=0.6, skip_box_thresh=0.1, conf_type="max")
    assert ensemble.iou_thresh == 0.6
    assert ensemble.skip_box_thresh == 0.1
    assert ensemble.conf_type == "max"
    assert len(ensemble.ensemble_models) == 2


def test_fuse_predictions_wbf(
    mock_models: list[DummyModel], sample_predictions: list[DetectionPrediction]
) -> None:
    """Test _fuse_predictions with WBF."""
    ensemble = WBFEnsemble(models=mock_models, iou_thresh=0.55)
    image_size = (1000, 1000)

    result = ensemble._fuse_predictions(sample_predictions, image_size)

    assert "boxes" in result
    assert "labels" in result
    assert "scores" in result

    assert isinstance(result["boxes"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)
    assert isinstance(result["scores"], torch.Tensor)

    # Check shape
    if result["boxes"].numel() > 0:
        assert result["boxes"].shape[1] == 4
        # Denormalization check
        assert torch.any(result["boxes"] > 1.0)


def test_fuse_predictions_empty(mock_models: list[DummyModel]) -> None:
    """Test _fuse_predictions with empty input."""
    ensemble = WBFEnsemble(models=mock_models)
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


def test_fuse_predictions_weights(
    mock_models: list[DummyModel], sample_predictions: list[DetectionPrediction]
) -> None:
    """Test WBF with model weights."""
    ensemble = WBFEnsemble(models=mock_models, weights=[0.9, 0.1])
    image_size = (100, 100)

    result = ensemble._fuse_predictions(sample_predictions, image_size)
    assert result["boxes"].numel() > 0
