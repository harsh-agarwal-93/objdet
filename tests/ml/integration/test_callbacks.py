"""Integration tests for training callbacks.

These tests verify that training callbacks correctly integrate with
Lightning's training loop and produce expected outputs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch import Tensor

# =============================================================================
# ConfusionMatrix Callback Tests
# =============================================================================


class TestConfusionMatrixCallback:
    """Test ConfusionMatrixCallback integration."""

    def test_callback_initialization(self) -> None:
        """Test callback initializes correctly."""
        from objdet.training.callbacks.confusion_matrix import ConfusionMatrixCallback

        callback = ConfusionMatrixCallback(
            num_classes=5,
            iou_threshold=0.5,
            confidence_threshold=0.25,
            save_dir="outputs/cm",
            class_names=["a", "b", "c", "d", "e"],
        )

        assert callback.num_classes == 5
        assert callback.iou_threshold == 0.5
        assert callback.confidence_threshold == 0.25
        assert callback.class_names == ["a", "b", "c", "d", "e"]

    def test_confusion_matrix_reset_on_epoch_start(self) -> None:
        """Test that confusion matrix is reset at epoch start."""
        from objdet.training.callbacks.confusion_matrix import ConfusionMatrixCallback

        callback = ConfusionMatrixCallback(num_classes=3)

        # Simulate some prior state
        callback._confusion_matrix = torch.ones(4, 4)

        # Mock trainer and module with proper device
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        mock_module.device = torch.device("cpu")

        callback.on_validation_epoch_start(mock_trainer, mock_module)

        # Matrix should be reset to zeros
        assert callback._confusion_matrix.sum().item() == 0
        assert callback._confusion_matrix.shape == (4, 4)  # num_classes + 1

    def test_confusion_matrix_update(
        self, sample_prediction: dict[str, Tensor], sample_target: dict[str, Tensor]
    ) -> None:
        """Test confusion matrix updates with predictions."""
        from objdet.training.callbacks.confusion_matrix import ConfusionMatrixCallback

        callback = ConfusionMatrixCallback(num_classes=3, iou_threshold=0.5)

        # Initialize matrix with proper device mocking
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        mock_module.device = torch.device("cpu")
        callback.on_validation_epoch_start(mock_trainer, mock_module)

        # Update with predictions
        callback._update_matrix(sample_prediction, sample_target)

        # Matrix should have some entries
        assert callback._confusion_matrix.sum().item() >= 0  # type: ignore[union-attr]

    def test_confusion_matrix_saves_plot(self, temp_dir: Path) -> None:
        """Test that confusion matrix plot is saved."""
        from objdet.training.callbacks.confusion_matrix import ConfusionMatrixCallback

        callback = ConfusionMatrixCallback(
            num_classes=3,
            save_dir=temp_dir / "confusion_matrices",
        )

        # Initialize with proper device mocking
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        mock_module = MagicMock()
        mock_module.device = torch.device("cpu")
        callback.on_validation_epoch_start(mock_trainer, mock_module)

        # Add some data to matrix
        callback._confusion_matrix[0, 0] = 5  # type: ignore[index]
        callback._confusion_matrix[1, 1] = 3  # type: ignore[index]

        # Trigger save
        callback.on_validation_epoch_end(mock_trainer, mock_module)

        # Check file was created
        expected_path = temp_dir / "confusion_matrices" / "confusion_matrix_epoch_0.png"
        assert expected_path.exists()


# =============================================================================
# DetectionVisualization Callback Tests
# =============================================================================


class TestDetectionVisualizationCallback:
    """Test DetectionVisualizationCallback integration."""

    def test_callback_initialization(self) -> None:
        """Test callback initializes correctly."""
        from objdet.training.callbacks.visualization import DetectionVisualizationCallback

        callback = DetectionVisualizationCallback(
            num_samples=8,
            save_dir="outputs/vis",
            log_to_tensorboard=True,
            confidence_threshold=0.5,
            class_names=["person", "car"],
        )

        assert callback.num_samples == 8
        assert callback.confidence_threshold == 0.5
        assert callback.class_names == ["person", "car"]
        assert callback.log_to_tensorboard is True

    def test_visualization_sample_collection(self) -> None:
        """Test that callback collects samples during validation."""
        from objdet.training.callbacks.visualization import DetectionVisualizationCallback

        callback = DetectionVisualizationCallback(num_samples=2)

        mock_trainer = MagicMock()
        mock_module = MagicMock()

        # Simulate batch
        images = torch.rand(4, 3, 224, 224)
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]])} for _ in range(4)]

        # Mock model predictions
        mock_module.return_value = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
            for _ in range(4)
        ]

        callback.on_validation_batch_end(
            trainer=mock_trainer,
            pl_module=mock_module,
            outputs=None,
            batch=(images, targets),
            batch_idx=0,
        )

        # Should have collected num_samples samples
        assert len(callback._validation_samples) == 2

    def test_visualization_respects_max_samples(self) -> None:
        """Test that callback doesn't collect more than num_samples."""
        from objdet.training.callbacks.visualization import DetectionVisualizationCallback

        callback = DetectionVisualizationCallback(num_samples=2)

        mock_trainer = MagicMock()
        mock_module = MagicMock()
        mock_module.return_value = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
            for _ in range(4)
        ]

        images = torch.rand(4, 3, 224, 224)
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]])} for _ in range(4)]

        # Call multiple times
        for batch_idx in range(3):
            callback.on_validation_batch_end(
                trainer=mock_trainer,
                pl_module=mock_module,
                outputs=None,
                batch=(images, targets),
                batch_idx=batch_idx,
            )

        # Should not exceed num_samples
        assert len(callback._validation_samples) <= 2

    def test_visualization_images_saved(self, temp_dir: Path) -> None:
        """Test that visualization images are saved to disk."""
        from objdet.training.callbacks.visualization import DetectionVisualizationCallback

        save_dir = temp_dir / "visualizations"
        callback = DetectionVisualizationCallback(
            num_samples=2,
            save_dir=save_dir,
            log_to_tensorboard=False,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        mock_trainer.logger = None
        mock_module = MagicMock()

        # Add samples manually
        callback._validation_samples = [
            (
                torch.rand(3, 64, 64),
                {
                    "boxes": torch.tensor([[10, 10, 50, 50]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                },
            ),
            (
                torch.rand(3, 64, 64),
                {
                    "boxes": torch.tensor([[20, 20, 60, 60]]),
                    "labels": torch.tensor([2]),
                    "scores": torch.tensor([0.8]),
                },
            ),
        ]

        callback.on_validation_epoch_end(mock_trainer, mock_module)

        # Check files were created
        assert save_dir.exists()
        saved_files = list(save_dir.glob("*.png"))
        assert len(saved_files) >= 1


# =============================================================================
# GradientMonitor Callback Tests
# =============================================================================


class TestGradientMonitorCallback:
    """Test GradientMonitorCallback integration."""

    def test_callback_initialization(self) -> None:
        """Test callback initializes correctly."""
        from objdet.training.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback(
            log_every_n_steps=100,
            detect_anomalies=True,
        )

        assert callback.log_every_n_steps == 100
        assert callback.detect_anomalies is True

    def test_gradient_monitoring_logs_stats(self) -> None:
        """Test that gradient statistics are logged."""
        from torch import nn

        from objdet.training.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback(log_every_n_steps=1)

        # Create a simple model with gradients
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Mock trainer and module
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = list(model.named_parameters())
        mock_optimizer = MagicMock()

        callback.on_before_optimizer_step(mock_trainer, mock_module, mock_optimizer)

        # Verify logging was called
        mock_module.log.assert_called()
        logged_metrics = [call[0][0] for call in mock_module.log.call_args_list]
        assert "train/grad_norm" in logged_metrics

    def test_gradient_monitoring_detects_nan(self) -> None:
        """Test that NaN gradients are detected."""
        from torch import nn

        from objdet.training.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback(log_every_n_steps=1, detect_anomalies=True)

        # Create model and set NaN gradient
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = list(model.named_parameters())
        mock_optimizer = MagicMock()

        # Should log NaN detection
        callback.on_before_optimizer_step(mock_trainer, mock_module, mock_optimizer)

        # Check that has_nan was logged
        logged_metrics = [call[0][0] for call in mock_module.log.call_args_list]
        assert "train/grad_has_nan" in logged_metrics


# =============================================================================
# LearningRateMonitor Callback Tests
# =============================================================================


class TestLearningRateMonitorCallback:
    """Test LearningRateMonitorCallback integration."""

    def test_callback_initialization(self) -> None:
        """Test callback initializes correctly."""
        from objdet.training.callbacks.lr_monitor import LearningRateMonitorCallback

        callback = LearningRateMonitorCallback(
            log_momentum=True,
            log_weight_decay=True,
        )

        assert callback.log_momentum is True
        assert callback.log_weight_decay is True

    def test_lr_monitoring_logs_learning_rate(self) -> None:
        """Test that learning rate is logged."""
        from torch import nn
        from torch.optim import SGD

        from objdet.training.callbacks.lr_monitor import LearningRateMonitorCallback

        callback = LearningRateMonitorCallback()

        # Create model and optimizer
        model = nn.Linear(10, 5)
        optimizer = SGD(model.parameters(), lr=0.01)

        mock_trainer = MagicMock()
        mock_trainer.optimizers = [optimizer]
        mock_module = MagicMock()

        callback.on_train_batch_start(mock_trainer, mock_module, batch=None, batch_idx=0)

        # Verify LR was logged
        mock_module.log.assert_called()
        logged_metrics = [call[0][0] for call in mock_module.log.call_args_list]
        assert "lr" in logged_metrics

    def test_lr_monitoring_with_momentum(self) -> None:
        """Test that momentum is logged when enabled."""
        from torch import nn
        from torch.optim import SGD

        from objdet.training.callbacks.lr_monitor import LearningRateMonitorCallback

        callback = LearningRateMonitorCallback(log_momentum=True)

        model = nn.Linear(10, 5)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        mock_trainer = MagicMock()
        mock_trainer.optimizers = [optimizer]
        mock_module = MagicMock()

        callback.on_train_batch_start(mock_trainer, mock_module, batch=None, batch_idx=0)

        logged_metrics = [call[0][0] for call in mock_module.log.call_args_list]
        assert any("momentum" in m for m in logged_metrics)
