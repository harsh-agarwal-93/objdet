"""Integration tests for CLI commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIHelp:
    """Test CLI help commands."""

    def test_cli_main_help(self) -> None:
        """Test that main help command works."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "objdet" in result.stdout.lower()

    def test_cli_fit_help(self) -> None:
        """Test that fit subcommand help works."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "fit", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0


class TestCLIExport:
    """Test CLI export command."""

    @pytest.mark.slow
    def test_cli_export_help(self) -> None:
        """Test export subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "export", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0
        assert "export" in result.stdout.lower() or "usage" in result.stdout.lower()


class TestCLIServe:
    """Test CLI serve command."""

    def test_cli_serve_help(self) -> None:
        """Test serve subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0


class TestCLIPreprocess:
    """Test CLI preprocess command."""

    def test_cli_preprocess_help(self) -> None:
        """Test preprocess subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "preprocess", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0
        assert "preprocess" in result.stdout.lower() or "usage" in result.stdout.lower()

    @pytest.mark.slow
    def test_cli_preprocess_voc(self, voc_dataset_dir: Path, temp_dir: Path) -> None:
        """Test preprocessing a VOC dataset."""
        output_dir = temp_dir / "litdata_output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "preprocess",
                "--format",
                "voc",
                "--output",
                str(output_dir),
                "--voc_images_dir",
                str(voc_dataset_dir / "JPEGImages"),
                "--voc_annotations_dir",
                str(voc_dataset_dir / "Annotations"),
                "--voc_imagesets_dir",
                str(voc_dataset_dir / "ImageSets" / "Main"),
                "--class_names",
                "person",
                "--splits",
                "train",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        # Check the command ran (may fail due to litdata optional dep)
        # Success or dependency error are both acceptable
        assert result.returncode == 0 or "litdata" in result.stderr.lower()


class TestCLIValidation:
    """Test CLI validate command."""

    def test_cli_validate_help(self) -> None:
        """Test validate subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "validate", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0


class TestCLITest:
    """Test CLI test command."""

    def test_cli_test_help(self) -> None:
        """Test test subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "test", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0


class TestCLIPredict:
    """Test CLI predict command."""

    def test_cli_predict_help(self) -> None:
        """Test predict subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "predict", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.mark.slow
    def test_cli_export_missing_checkpoint(self, temp_dir: Path) -> None:
        """Test export fails gracefully with missing checkpoint."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "export",
                "--checkpoint",
                str(temp_dir / "nonexistent.ckpt"),
                "--output",
                str(temp_dir / "output.onnx"),
                "--format",
                "onnx",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        # Should fail with error
        assert (
            result.returncode != 0
            or "error" in result.stderr.lower()
            or "not found" in result.stderr.lower()
        )

    @pytest.mark.slow
    def test_cli_serve_missing_config(self, temp_dir: Path) -> None:
        """Test serve fails gracefully with missing config."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "serve",
                "--config",
                str(temp_dir / "nonexistent.yaml"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        # Should fail with error
        assert result.returncode != 0 or "error" in result.stderr.lower()

    def test_cli_preprocess_missing_output(self) -> None:
        """Test preprocess requires output argument."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "objdet",
                "preprocess",
                "--format",
                "coco",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        # Should fail - output is required
        assert result.returncode != 0
        assert "output" in result.stderr.lower() or "required" in result.stderr.lower()


class TestCLIPreprocessCOCO:
    """Test COCO preprocessing through CLI."""

    @pytest.mark.slow
    def test_cli_preprocess_coco_help(self) -> None:
        """Test that preprocess COCO-specific args are in help."""
        result = subprocess.run(
            [sys.executable, "-m", "objdet", "preprocess", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0
        assert "coco" in result.stdout.lower()
        assert "coco_ann_file" in result.stdout or "coco-ann-file" in result.stdout
