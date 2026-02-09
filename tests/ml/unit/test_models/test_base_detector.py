"""Unit tests for BaseLightningDetector."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from objdet.core.constants import ClassIndexMode
from objdet.core.types import DetectionPrediction, DetectionTarget
from objdet.models.base import BaseLightningDetector


class MockDetector(BaseLightningDetector):
    """Subclass of BaseLightningDetector for testing."""

    def _build_model(self) -> nn.Module:
        """Create a simple linear layer."""
        return nn.Linear(1, 1)

    def forward(
        self,
        images: list[Tensor],
        targets: list[DetectionTarget] | None = None,
    ) -> dict[str, Tensor] | list[DetectionPrediction]:
        """Mock forward pass."""
        if targets is not None:
            # Training mode: return losses
            return {
                "loss_box": torch.tensor(0.5, requires_grad=True),
                "loss_cls": torch.tensor(0.5, requires_grad=True),
            }

        # Inference mode: return predictions
        return [
            cast(
                "DetectionPrediction",
                {
                    "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                },
            )
            for _ in images
        ]


@pytest.fixture
def model() -> MockDetector:
    """Create a MockDetector instance."""
    return MockDetector(
        num_classes=3,
        class_index_mode=ClassIndexMode.TORCHVISION,
        learning_rate=0.01,
    )


def test_init(model: MockDetector) -> None:
    """Test initialization and hyperparameter saving."""
    assert model.num_classes == 3
    assert model.class_index_mode == ClassIndexMode.TORCHVISION
    assert model.learning_rate == pytest.approx(0.01)
    assert isinstance(model.model, nn.Module)
    assert model.num_model_classes == 4  # TorchVision mode adds 1 for background


def test_init_yolo() -> None:
    """Test initialization in YOLO mode."""
    model = MockDetector(num_classes=3, class_index_mode=ClassIndexMode.YOLO)
    assert model.num_model_classes == 3  # YOLO mode has no background class


def test_init_string_mode() -> None:
    """Test initialization with string mode."""
    model = MockDetector(num_classes=3, class_index_mode="yolo")
    assert model.class_index_mode == ClassIndexMode.YOLO


def test_training_step(model: MockDetector) -> None:
    """Test a single training step."""
    batch = (
        [torch.randn(3, 224, 224)],
        [
            cast(
                "DetectionTarget",
                {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
            )
        ],
    )
    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() == pytest.approx(1.0)  # 0.5 + 0.5
    assert loss.requires_grad


def test_validation_step(model: MockDetector) -> None:
    """Test a single validation step."""
    batch = (
        [torch.randn(3, 224, 224)],
        [
            cast(
                "DetectionTarget",
                {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
            )
        ],
    )

    # Validation step updates metrics
    model.validation_step(batch, 0)

    # Check metrics
    metrics = model._val_map.compute()
    assert "map" in metrics
    assert metrics["map"] >= 0


def test_on_validation_epoch_end(model: MockDetector) -> None:
    """Test validation epoch end logic."""
    # Mock some metrics
    model._val_map.compute = lambda: {
        "map": torch.tensor(0.5),
        "map_50": torch.tensor(0.8),
        "map_75": torch.tensor(0.4),
    }

    # This shouldn't crash
    model.on_validation_epoch_end()


def test_test_step(model: MockDetector) -> None:
    """Test a single test step."""
    batch = (
        [torch.randn(3, 224, 224)],
        [
            cast(
                "DetectionTarget",
                {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
            )
        ],
    )
    model.test_step(batch, 0)

    metrics = model._test_map.compute()
    assert "map" in metrics


def test_on_test_epoch_end(model: MockDetector) -> None:
    """Test test epoch end logic."""
    model._test_map.compute = lambda: {
        "map": torch.tensor(0.5),
        "map_50": torch.tensor(0.8),
        "map_75": torch.tensor(0.4),
    }
    model.on_test_epoch_end()


def test_predict_step(model: MockDetector) -> None:
    """Test a single predict step."""
    # List interface
    batch_list = [torch.randn(3, 224, 224)]
    preds = model.predict_step(batch_list, 0)
    assert len(preds) == 1
    assert "boxes" in preds[0]

    # Tuple interface
    batch_tuple = ([torch.randn(3, 224, 224)], None)
    preds = model.predict_step(batch_tuple, 0)
    assert len(preds) == 1


def test_filter_predictions(model: MockDetector) -> None:
    """Test prediction filtering logic."""
    raw_preds: list[DetectionPrediction] = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.9, 0.1]),  # One above, one below default threshold (0.5)
        }
    ]

    filtered = model._filter_predictions(raw_preds)
    assert len(filtered[0]["boxes"]) == 1
    assert filtered[0]["scores"][0] == pytest.approx(0.9)

    # Test empty filtering
    empty_raw: list[DetectionPrediction] = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.1]),
        }
    ]
    filtered_empty = model._filter_predictions(empty_raw)
    assert len(filtered_empty[0]["boxes"]) == 0


def test_configure_optimizers(model: MockDetector) -> None:
    """Test optimizer configuration."""
    # Test AdamW
    config = model.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.AdamW)
    assert "lr_scheduler" in config

    # SGD
    model.optimizer_name = "sgd"
    config_sgd = model.configure_optimizers()
    assert isinstance(config_sgd["optimizer"], torch.optim.SGD)

    # Unknown
    model.optimizer_name = "unknown"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        model.configure_optimizers()


def test_build_scheduler(model: MockDetector) -> None:
    """Test different scheduler configurations."""
    optimizer = torch.optim.AdamW(model.parameters())

    # Cosine
    model.scheduler_name = "cosine"
    sched = model._build_scheduler(optimizer)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    # Step
    model.scheduler_name = "step"
    sched = model._build_scheduler(optimizer)
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    # Plateau
    model.scheduler_name = "plateau"
    sched = model._build_scheduler(optimizer)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    # Unknown
    model.scheduler_name = "unknown"
    with pytest.raises(ValueError, match="Unknown scheduler"):
        model._build_scheduler(optimizer)


def test_get_model_info(model: MockDetector) -> None:
    """Test model metadata retrieval."""
    info = model.get_model_info()
    assert info["model_type"] == "MockDetector"
    assert info["num_classes"] == 3
    assert info["class_index_mode"] == ClassIndexMode.TORCHVISION.value
    assert "num_parameters" in info
    assert "trainable_parameters" in info
