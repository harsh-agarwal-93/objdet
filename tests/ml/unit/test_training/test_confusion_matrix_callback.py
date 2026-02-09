"""Unit tests for ConfusionMatrixCallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch

from objdet.training.callbacks.confusion_matrix import ConfusionMatrixCallback


@pytest.fixture
def callback(tmp_path: Path) -> ConfusionMatrixCallback:
    """Create a ConfusionMatrixCallback instance."""
    return ConfusionMatrixCallback(
        num_classes=3,
        save_dir=tmp_path / "cm",
        class_names=["cat", "dog", "bird"],
    )


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Create a mock trainer."""
    trainer = MagicMock(spec=L.Trainer)
    trainer.current_epoch = 0
    return trainer


@pytest.fixture
def mock_pl_module() -> MagicMock:
    """Create a mock lightning module."""
    model = MagicMock(spec=L.LightningModule)
    model.device = torch.device("cpu")
    return model


def test_callback_init(tmp_path: Path) -> None:
    """Test callback initialization."""
    cb = ConfusionMatrixCallback(num_classes=10, save_dir=tmp_path)
    assert cb.num_classes == 10
    assert cb.save_dir == tmp_path
    assert cb.iou_threshold == 0.5
    assert cb._confusion_matrix is None


def test_on_validation_epoch_start(
    callback: ConfusionMatrixCallback, mock_trainer: MagicMock, mock_pl_module: MagicMock
) -> None:
    """Test matrix initialization at epoch start."""
    callback.on_validation_epoch_start(mock_trainer, mock_pl_module)
    assert callback._confusion_matrix is not None
    assert callback._confusion_matrix.shape == (4, 4)  # num_classes + 1
    assert torch.all(callback._confusion_matrix == 0)


def test_update_matrix_basic(callback: ConfusionMatrixCallback) -> None:
    """Test _update_matrix with basic predictions."""
    callback._confusion_matrix = torch.zeros((4, 4))

    pred = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "labels": torch.tensor([0]),
        "scores": torch.tensor([0.9]),
    }
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "labels": torch.tensor([0]),
    }

    callback._update_matrix(pred, target)

    # [0, 0] should be 1 (Cat matched as Cat)
    assert callback._confusion_matrix[0, 0] == 1


def test_update_matrix_false_positive(callback: ConfusionMatrixCallback) -> None:
    """Test _update_matrix with a false positive (background match)."""
    callback._confusion_matrix = torch.zeros((4, 4))

    pred = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }
    target = {
        "boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0]]),  # No overlap
        "labels": torch.tensor([1]),
    }

    callback._update_matrix(pred, target)

    # Pred class 1, True class 3 (Background)
    assert callback._confusion_matrix[3, 1] == 1
    # True class 1, Pred class 3 (False negative)
    assert callback._confusion_matrix[1, 3] == 1


def test_compute_iou(callback: ConfusionMatrixCallback) -> None:
    """Test IoU computation logic."""
    box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
    box2 = torch.tensor([5.0, 5.0, 15.0, 15.0])

    iou = callback._compute_iou(box1, box2)
    # Verify intersection area logic
    assert pytest.approx(iou) == 1 / 7


def test_on_validation_batch_end(
    callback: ConfusionMatrixCallback, mock_trainer: MagicMock, mock_pl_module: MagicMock
) -> None:
    """Test on_validation_batch_end integration."""
    callback._confusion_matrix = torch.zeros((4, 4))

    batch = (
        torch.zeros((1, 3, 100, 100)),  # Images
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}],  # Targets
    )

    mock_pl_module.return_value = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.9]),
        }
    ]

    callback.on_validation_batch_end(
        mock_trainer, mock_pl_module, outputs=None, batch=batch, batch_idx=0
    )

    assert callback._confusion_matrix[0, 0] == 1


@patch("torch.save")
@patch("matplotlib.pyplot.savefig")
def test_on_validation_epoch_end(
    mock_savefig: MagicMock,
    mock_save: MagicMock,
    callback: ConfusionMatrixCallback,
    mock_trainer: MagicMock,
    mock_pl_module: MagicMock,
) -> None:
    """Test epoch end saving logic."""
    callback._confusion_matrix = torch.zeros((4, 4))
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    assert mock_save.called
    assert mock_savefig.called


def test_on_validation_epoch_end_no_matrix(
    callback: ConfusionMatrixCallback, mock_trainer: MagicMock, mock_pl_module: MagicMock
) -> None:
    """Test epoch end when matrix is not initialized."""
    callback._confusion_matrix = None
    # Should return early without error
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)


def test_update_matrix_gt_matched_already(callback: ConfusionMatrixCallback) -> None:
    """Test matching when GT is already matched."""
    callback._confusion_matrix = torch.zeros((4, 4))

    # Two identical predictions, only one should match the single GT
    pred = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0]]),
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([0.9, 0.9]),
    }
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
        "labels": torch.tensor([0]),
    }

    callback._update_matrix(pred, target)

    # TP=1, FP=1 (matched as background)
    assert callback._confusion_matrix[0, 0] == 1  # TP
    assert callback._confusion_matrix[3, 0] == 1  # FP (BG vs class 0)


def test_compute_iou_zero_union(callback: ConfusionMatrixCallback) -> None:
    """Test IoU with zero union area."""
    box1 = torch.zeros(4)
    box2 = torch.zeros(4)
    import math

    assert math.isclose(callback._compute_iou(box1, box2), 0.0)


def test_save_plot_normalization(callback: ConfusionMatrixCallback, tmp_path: Path) -> None:
    """Test different normalization modes in _save_plot."""
    callback._confusion_matrix = torch.ones((4, 4))
    callback.save_dir.mkdir(parents=True, exist_ok=True)

    # Test "pred"
    callback.normalize = "pred"
    callback._save_plot(0)

    # Test "all"
    callback.normalize = "all"
    callback._save_plot(0)

    # Test None
    callback.normalize = None
    callback._save_plot(0)


def test_save_plot_no_class_names(tmp_path: Path) -> None:
    """Test plot saving without explicit class names."""
    cb = ConfusionMatrixCallback(num_classes=3, save_dir=tmp_path)
    cb._confusion_matrix = torch.zeros((4, 4))
    cb._save_plot(0)


@patch("objdet.training.callbacks.confusion_matrix.logger")
def test_on_validation_epoch_end_import_error(
    mock_logger: MagicMock,
    callback: ConfusionMatrixCallback,
    mock_trainer: MagicMock,
    mock_pl_module: MagicMock,
) -> None:
    """Test handling of matplotlib ImportError."""
    callback._confusion_matrix = torch.zeros((4, 4))

    # Force ImportError in _save_plot
    with patch(
        "objdet.training.callbacks.confusion_matrix.ConfusionMatrixCallback._save_plot",
        side_effect=ImportError,
    ):
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
        assert mock_logger.warning.called
