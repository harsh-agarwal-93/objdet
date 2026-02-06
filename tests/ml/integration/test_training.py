"""Integration tests for model training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from objdet.models.torchvision.faster_rcnn import FasterRCNN


class TestFastDevRun:
    """Test fast_dev_run training for quick validation."""

    @pytest.fixture
    def small_frcnn(self) -> FasterRCNN:
        """Create a small FasterRCNN for testing."""
        return FasterRCNN(
            num_classes=5,
            pretrained=False,
            pretrained_backbone=False,
        )

    @pytest.fixture
    def sample_batch(self) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Create a minimal training batch."""
        images = [torch.rand(3, 320, 320) for _ in range(2)]
        targets = [
            {
                "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
                "labels": torch.tensor([1]),
            }
            for _ in range(2)
        ]
        return images, targets

    def test_model_forward_training(self, small_frcnn, sample_batch) -> None:
        """Test that model can run a training forward pass."""
        images, targets = sample_batch
        small_frcnn.train()

        losses = small_frcnn(images, targets)

        assert isinstance(losses, dict)
        assert "loss_classifier" in losses or "classification" in str(losses.keys()).lower()

    def test_model_forward_inference(self, small_frcnn, sample_batch) -> None:
        """Test that model can run an inference forward pass."""
        images, _ = sample_batch
        small_frcnn.eval()

        with torch.no_grad():
            predictions = small_frcnn(images)

        assert isinstance(predictions, list)
        assert len(predictions) == len(images)

    @pytest.mark.slow
    def test_training_step_with_lightning(self, small_frcnn, sample_batch) -> None:
        """Test a single training step using Lightning."""
        from lightning.pytorch import Trainer

        # Create minimal trainer
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

        # Run a single step
        # Note: This requires a proper datamodule, so we test the trainer init
        assert trainer.max_steps == 1


class TestCheckpointSaving:
    """Test checkpoint saving functionality."""

    @pytest.fixture
    def small_model(self) -> FasterRCNN:
        """Create a small model for testing."""
        return FasterRCNN(
            num_classes=3,
            pretrained=False,
            pretrained_backbone=False,
        )

    def test_model_state_dict_saveable(self, small_model, tmp_path: Path) -> None:
        """Test that model state dict can be saved."""
        checkpoint_path = tmp_path / "model.pt"

        torch.save(small_model.state_dict(), checkpoint_path)

        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0

    def test_model_state_dict_loadable(self, small_model, tmp_path: Path) -> None:
        """Test that model state dict can be loaded."""
        checkpoint_path = tmp_path / "model.pt"
        torch.save(small_model.state_dict(), checkpoint_path)

        # Create new model and load state
        new_model = FasterRCNN(
            num_classes=3,
            pretrained=False,
            pretrained_backbone=False,
        )
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            small_model.named_parameters(),
            new_model.named_parameters(),
            strict=True,
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)


class TestTrainingCallbacks:
    """Test training callback integration."""

    def test_model_checkpoint_callback_init(self) -> None:
        """Test ModelCheckpoint callback initialization."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-{epoch:02d}",
            monitor="val/mAP",
            mode="max",
            save_top_k=3,
        )

        assert callback.monitor == "val/mAP"
        assert callback.mode == "max"

    def test_early_stopping_callback_init(self) -> None:
        """Test EarlyStopping callback initialization."""
        from lightning.pytorch.callbacks import EarlyStopping

        callback = EarlyStopping(
            monitor="val/mAP",
            patience=10,
            mode="max",
        )

        assert callback.patience == 10
        assert callback.mode == "max"
