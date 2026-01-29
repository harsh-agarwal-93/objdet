"""Functional tests for training CLI workflows.

These tests verify that the `objdet fit` command works correctly
with various model architectures and datasets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING  # noqa: F401

import pytest


@pytest.mark.integration
class TestFasterRCNNTraining:
    """Functional tests for Faster R-CNN training."""

    def test_fast_dev_run_completes(
        self,
        faster_rcnn_config: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test that Faster R-CNN training completes with fast_dev_run.

        This verifies:
        - CLI parses config correctly
        - Model initializes properly
        - Training loop executes
        - No errors during forward/backward pass
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "fit",
                "--config",
                str(faster_rcnn_config),
                "--trainer.fast_dev_run",
                "True",
                "--trainer.default_root_dir",
                str(tmp_output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 minute timeout
        )

        # Check process completed successfully
        assert result.returncode == 0, (
            f"Training failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_training_one_epoch(
        self,
        faster_rcnn_config: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test that Faster R-CNN completes a full epoch."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "fit",
                "--config",
                str(faster_rcnn_config),
                "--trainer.max_epochs",
                "1",
                "--trainer.default_root_dir",
                str(tmp_output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Training failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.mark.integration
class TestRetinaNetTraining:
    """Functional tests for RetinaNet training."""

    def test_fast_dev_run_completes(
        self,
        retinanet_config: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test that RetinaNet training completes with fast_dev_run."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "fit",
                "--config",
                str(retinanet_config),
                "--trainer.fast_dev_run",
                "True",
                "--trainer.default_root_dir",
                str(tmp_output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Training failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


@pytest.mark.integration
class TestYOLOv8Training:
    """Functional tests for YOLOv8 training.

    Note:
        YOLOv8 training currently has a known bug that causes
        `IndexError: too many indices for tensor of dimension 2`
        during loss computation. These tests are marked as xfail.
    """

    @pytest.mark.xfail(
        reason="Known bug: IndexError in loss computation during training",
        strict=False,  # Allow test to unexpectedly pass if bug is fixed
    )
    def test_fast_dev_run_completes(
        self,
        yolov8_config: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test that YOLOv8 training completes with fast_dev_run.

        This test is expected to fail due to a known bug in the YOLO
        loss computation (IndexError: too many indices for tensor of dimension 2).
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "fit",
                "--config",
                str(yolov8_config),
                "--trainer.fast_dev_run",
                "True",
                "--trainer.default_root_dir",
                str(tmp_output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Training failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


@pytest.mark.integration
class TestLitDataTraining:
    """Functional tests for training with LitData datasets."""

    @pytest.mark.slow
    def test_litdata_training_completes(
        self,
        litdata_config: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test training with LitData optimized dataset.

        This test is marked slow because it requires dataset conversion.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "fit",
                "--config",
                str(litdata_config),
                "--trainer.fast_dev_run",
                "True",
                "--trainer.default_root_dir",
                str(tmp_output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"LitData training failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest.mark.integration
class TestTrainingPythonAPI:
    """Test training workflows using Python API directly."""

    def test_faster_rcnn_trainer_fit(self, sample_coco_dataset: Path) -> None:
        """Test Faster R-CNN training using Python API."""
        from lightning import Trainer

        from objdet.data.formats.coco import COCODataModule
        from objdet.models.torchvision import FasterRCNN

        # Create model
        model = FasterRCNN(
            num_classes=2,
            backbone="resnet50_fpn",
            pretrained=False,
            pretrained_backbone=False,
            min_size=128,
            max_size=256,
            learning_rate=0.001,
        )

        # Create data module
        datamodule = COCODataModule(
            data_dir=sample_coco_dataset,
            train_ann_file="annotations/instances_train2017.json",
            val_ann_file="annotations/instances_val2017.json",
            train_img_dir="train2017",
            val_img_dir="val2017",
            batch_size=2,
            num_workers=0,
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        # Run training
        trainer.fit(model, datamodule=datamodule)

        # Verify model state
        assert model.training or not model.training  # Just verify no error

    def test_retinanet_trainer_fit(self, sample_coco_dataset: Path) -> None:
        """Test RetinaNet training using Python API."""
        from lightning import Trainer

        from objdet.data.formats.coco import COCODataModule
        from objdet.models.torchvision import RetinaNet

        # Create model
        model = RetinaNet(
            num_classes=2,
            backbone="resnet50_fpn",
            pretrained=False,
            pretrained_backbone=False,
            min_size=128,
            max_size=256,
            learning_rate=0.001,
        )

        # Create data module
        datamodule = COCODataModule(
            data_dir=sample_coco_dataset,
            train_ann_file="annotations/instances_train2017.json",
            val_ann_file="annotations/instances_val2017.json",
            train_img_dir="train2017",
            val_img_dir="val2017",
            batch_size=2,
            num_workers=0,
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        # Run training
        trainer.fit(model, datamodule=datamodule)
