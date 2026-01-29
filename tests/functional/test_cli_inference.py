"""Functional tests for inference CLI workflows.

These tests verify that the `objdet validate`, `objdet test`,
and `objdet predict` commands work correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING  # noqa: F401

import pytest
import torch


@pytest.mark.integration
class TestValidationWorkflow:
    """Functional tests for validation workflow."""

    def test_validate_with_config(
        self,
        faster_rcnn_config: Path,
        tmp_output_dir: Path,
    ) -> None:
        """Test that validation runs without errors.

        Note: This runs validation without a checkpoint, which tests
        the data loading and forward pass only.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "validate",
                "--config",
                str(faster_rcnn_config),
                "--trainer.limit_val_batches",
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
            f"Validation failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


@pytest.mark.integration
class TestTestWorkflow:
    """Functional tests for test workflow."""

    def test_test_with_config(
        self,
        faster_rcnn_config: Path,
        sample_coco_dataset: Path,
        tmp_output_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test that test command runs without errors.

        Uses validation data as test data for this test.
        """
        import yaml

        # Modify config to add test annotations
        with open(faster_rcnn_config) as f:
            config = yaml.safe_load(f)

        # Use val set as test set
        config["data"]["init_args"]["test_ann_file"] = "annotations/instances_val2017.json"
        config["data"]["init_args"]["test_img_dir"] = "val2017"

        test_config = tmp_path / "test_config.yaml"
        with open(test_config, "w") as f:
            yaml.dump(config, f)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "test",
                "--config",
                str(test_config),
                "--trainer.limit_test_batches",
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
            f"Test failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


@pytest.mark.integration
class TestPredictionWorkflow:
    """Functional tests for prediction workflow."""

    def test_predict_python_api(
        self,
        sample_coco_dataset: Path,
    ) -> None:
        """Test prediction using Python API directly."""
        from objdet.inference.predictor import Predictor
        from objdet.models.torchvision import FasterRCNN

        # Create model
        model = FasterRCNN(
            num_classes=2,
            backbone="resnet50_fpn",
            pretrained=False,
            pretrained_backbone=False,
            min_size=128,
            max_size=256,
        )

        # Create predictor
        predictor = Predictor(
            model=model,
            device="cpu",
            confidence_threshold=0.1,
        )

        # Load a test image
        image_path = sample_coco_dataset / "train2017" / "image_0001.jpg"
        assert image_path.exists(), f"Test image not found: {image_path}"

        from typing import cast

        # Run prediction
        result = predictor.predict(str(image_path))

        # Verify result structure
        result = cast("dict", result)
        assert "boxes" in result
        assert "labels" in result
        assert "scores" in result
        assert isinstance(result["boxes"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)
        assert isinstance(result["scores"], torch.Tensor)

    def test_predict_batch(
        self,
        sample_coco_dataset: Path,
    ) -> None:
        """Test batch prediction using Python API."""
        from objdet.inference.predictor import Predictor
        from objdet.models.torchvision import FasterRCNN

        # Create model
        model = FasterRCNN(
            num_classes=2,
            backbone="resnet50_fpn",
            pretrained=False,
            pretrained_backbone=False,
            min_size=128,
            max_size=256,
        )

        # Create predictor
        predictor = Predictor(
            model=model,
            device="cpu",
            confidence_threshold=0.1,
        )

        # Load test images
        image_paths: list[str | Path | torch.Tensor] = [
            str(sample_coco_dataset / "train2017" / f"image_{i:04d}.jpg") for i in range(1, 4)
        ]

        # Run batch prediction
        results = predictor.predict_batch(image_paths, batch_size=2)

        # Verify results
        assert len(results) == 3
        for result in results:
            assert "boxes" in result
            assert "labels" in result
            assert "scores" in result


@pytest.mark.integration
class TestValidationPythonAPI:
    """Test validation workflow using Python API directly."""

    def test_faster_rcnn_validation(self, sample_coco_dataset: Path) -> None:
        """Test Faster R-CNN validation using Python API."""
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
            accelerator="cpu",
            devices=1,
            limit_val_batches=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        # Run validation
        results = trainer.validate(model, datamodule=datamodule)

        # Verify we got results
        assert isinstance(results, list)
        assert len(results) > 0

    def test_retinanet_validation(self, sample_coco_dataset: Path) -> None:
        """Test RetinaNet validation using Python API."""
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
            accelerator="cpu",
            devices=1,
            limit_val_batches=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        # Run validation
        results = trainer.validate(model, datamodule=datamodule)

        # Verify we got results
        assert isinstance(results, list)
        assert len(results) > 0
