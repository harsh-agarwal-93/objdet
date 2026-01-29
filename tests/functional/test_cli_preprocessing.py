"""Functional tests for preprocessing CLI workflows.

These tests verify that the `objdet preprocess` command correctly
converts datasets to LitData format.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING  # noqa: F401

import pytest


@pytest.mark.integration
class TestCOCOToLitDataConversion:
    """Functional tests for COCO to LitData conversion."""

    def test_preprocess_coco_cli(
        self,
        sample_coco_dataset: Path,
        tmp_path: Path,
    ) -> None:
        """Test COCO to LitData conversion via CLI.

        Verifies:
        - CLI parses arguments correctly
        - Conversion runs without errors
        - Output directory is created
        """
        output_dir = tmp_path / "litdata_output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "preprocess",
                "--format",
                "coco",
                "--coco_ann_file",
                str(sample_coco_dataset / "annotations" / "instances_train2017.json"),
                "--coco_images_dir",
                str(sample_coco_dataset / "train2017"),
                "--output",
                str(output_dir / "train"),
                "--num_workers",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Preprocessing failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify output was created
        assert (output_dir / "train").exists(), "LitData output directory was not created"

    def test_preprocess_coco_python_api(
        self,
        sample_coco_dataset: Path,
        tmp_path: Path,
    ) -> None:
        """Test COCO to LitData conversion via Python API."""
        from objdet.data.preprocessing import convert_to_litdata

        output_dir = tmp_path / "litdata_output"

        # Convert train split
        convert_to_litdata(
            output_dir=str(output_dir),
            format_name="coco",
            coco_ann_file=str(sample_coco_dataset / "annotations" / "instances_train2017.json"),
            coco_images_dir=str(sample_coco_dataset / "train2017"),
            splits=["train"],
            num_workers=1,
        )

        # Verify output was created
        train_dir = output_dir / "train"
        assert train_dir.exists(), "Train split directory was not created"

        # Check for LitData files
        litdata_files = list(train_dir.glob("*.bin")) + list(train_dir.glob("index.json"))
        assert len(litdata_files) > 0, "No LitData files were created"


@pytest.mark.integration
class TestMultipleSplitsConversion:
    """Test conversion of multiple splits."""

    def test_preprocess_train_and_val(
        self,
        sample_coco_dataset: Path,
        tmp_path: Path,
    ) -> None:
        """Test converting both train and val splits."""
        from objdet.data.preprocessing import convert_to_litdata

        output_dir = tmp_path / "litdata_output"

        # Convert train split
        convert_to_litdata(
            output_dir=str(output_dir),
            format_name="coco",
            coco_ann_file=str(sample_coco_dataset / "annotations" / "instances_train2017.json"),
            coco_images_dir=str(sample_coco_dataset / "train2017"),
            splits=["train"],
            num_workers=1,
        )

        # Convert val split
        convert_to_litdata(
            output_dir=str(output_dir),
            format_name="coco",
            coco_ann_file=str(sample_coco_dataset / "annotations" / "instances_val2017.json"),
            coco_images_dir=str(sample_coco_dataset / "val2017"),
            splits=["val"],
            num_workers=1,
        )

        # Verify both splits exist
        assert (output_dir / "train").exists(), "Train split not created"
        assert (output_dir / "val").exists(), "Val split not created"


@pytest.mark.integration
class TestLitDataDataModuleLoading:
    """Test that converted LitData can be loaded."""

    @pytest.mark.slow
    def test_load_converted_dataset(
        self,
        sample_litdata_dataset: Path,
    ) -> None:
        """Test that converted LitData dataset can be loaded.

        This uses the sample_litdata_dataset fixture which
        automatically converts the sample COCO data.
        """
        from objdet.data.formats.litdata import LitDataDataModule

        # Create data module
        datamodule = LitDataDataModule(
            data_dir=sample_litdata_dataset,
            train_subdir="train",
            val_subdir="val",
            batch_size=2,
            num_workers=0,
        )

        # Setup
        datamodule.setup("fit")

        # Get train dataloader and check we can iterate
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Verify batch structure
        assert isinstance(batch, tuple), "Batch should be a tuple"
        assert len(batch) == 2, "Batch should contain (images, targets)"

        images, targets = batch
        assert len(images) > 0, "Should have at least one image"
        assert len(targets) > 0, "Should have at least one target"

        # Check image format
        assert images[0].dim() == 3, "Image should be 3D (C, H, W)"
        assert images[0].shape[0] == 3, "Image should have 3 channels"

        # Check target format
        assert "boxes" in targets[0], "Target should have boxes"
        assert "labels" in targets[0], "Target should have labels"


@pytest.mark.integration
class TestPreprocessingWithInput:
    """Test preprocessing with input directory specified."""

    def test_preprocess_with_input_dir(
        self,
        sample_coco_dataset: Path,
        tmp_path: Path,
    ) -> None:
        """Test preprocessing when input directory is specified."""
        # The sample_coco_dataset uses standard COCO structure
        # so we can test with --input

        output_dir = tmp_path / "litdata_output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "preprocess",
                "--input",
                str(sample_coco_dataset),
                "--format",
                "coco",
                "--output",
                str(output_dir / "train"),
                "--coco_ann_file",
                str(sample_coco_dataset / "annotations" / "instances_train2017.json"),
                "--coco_images_dir",
                str(sample_coco_dataset / "train2017"),
                "--num_workers",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Preprocessing failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
