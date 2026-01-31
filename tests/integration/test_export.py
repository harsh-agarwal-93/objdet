"""Integration tests for model export functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from objdet.models.torchvision.faster_rcnn import FasterRCNN
from objdet.optimization.export import (
    export_to_onnx,
    export_to_safetensors,
)


@pytest.fixture
def small_model() -> FasterRCNN:
    """Create a small FasterRCNN model for testing."""
    model = FasterRCNN(
        num_classes=5,
        pretrained=False,
        pretrained_backbone=False,
    )
    model.eval()
    return model


class TestExportToONNX:
    """Test ONNX export functionality.

    Note: These tests are marked as xfail because Faster R-CNN (and other
    detection models with NMS) have data-dependent control flow that
    torch.export cannot trace. This is a known PyTorch/Torchvision limitation:
    - NMS outputs variable-length tensors based on input data
    - torch.export requires static shapes for tracing
    - See: https://pytorch.org/docs/stable/generated/exportdb/index.html

    To export detection models to ONNX, consider using torch.jit.trace
    or the legacy torch.onnx.export with dynamo=False.
    """

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Faster R-CNN NMS has data-dependent shapes that torch.export cannot trace",
        raises=Exception,
    )
    def test_export_to_onnx_basic(self, small_model: FasterRCNN, temp_dir: Path) -> None:
        """Test basic ONNX export."""
        output_path = temp_dir / "model.onnx"

        result = export_to_onnx(
            model=small_model,
            output_path=output_path,
            input_shape=(1, 3, 224, 224),
            simplify=False,  # Skip simplification for speed
        )

        assert result.exists()
        assert result.suffix == ".onnx"
        assert result.stat().st_size > 0

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Faster R-CNN NMS has data-dependent shapes that torch.export cannot trace",
        raises=Exception,
    )
    def test_export_to_onnx_with_simplify(self, small_model: FasterRCNN, temp_dir: Path) -> None:
        """Test ONNX export with simplification."""
        output_path = temp_dir / "model_simplified.onnx"

        try:
            result = export_to_onnx(
                model=small_model,
                output_path=output_path,
                input_shape=(1, 3, 320, 320),
                simplify=True,
            )
            assert result.exists()
        except ImportError:
            # onnxsim may not be installed
            pytest.skip("onnxsim not installed")

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Faster R-CNN NMS has data-dependent shapes that torch.export cannot trace",
        raises=Exception,
    )
    def test_export_to_onnx_custom_opset(self, small_model: FasterRCNN, temp_dir: Path) -> None:
        """Test ONNX export with custom opset version."""
        output_path = temp_dir / "model_opset14.onnx"

        result = export_to_onnx(
            model=small_model,
            output_path=output_path,
            input_shape=(1, 3, 224, 224),
            opset_version=14,
            simplify=False,
        )

        assert result.exists()


class TestExportToSafetensors:
    """Test SafeTensors export functionality."""

    def test_export_to_safetensors_basic(self, small_model: FasterRCNN, temp_dir: Path) -> None:
        """Test basic SafeTensors export."""
        output_path = temp_dir / "model.safetensors"

        result = export_to_safetensors(
            model=small_model,
            output_path=output_path,
            include_metadata=True,
        )

        assert result.exists()
        assert result.suffix == ".safetensors"
        assert result.stat().st_size > 0

    def test_export_to_safetensors_without_metadata(
        self, small_model: FasterRCNN, temp_dir: Path
    ) -> None:
        """Test SafeTensors export without metadata."""
        output_path = temp_dir / "model_no_meta.safetensors"

        result = export_to_safetensors(
            model=small_model,
            output_path=output_path,
            include_metadata=False,
        )

        assert result.exists()

    def test_safetensors_can_reload(self, small_model: FasterRCNN, temp_dir: Path) -> None:
        """Test that exported SafeTensors can be reloaded."""
        output_path = temp_dir / "model_reload.safetensors"

        export_to_safetensors(
            model=small_model,
            output_path=output_path,
        )

        # Try to load the safetensors file
        try:
            from safetensors.torch import load_file

            state_dict = load_file(output_path)
            assert len(state_dict) > 0
        except ImportError:
            pytest.skip("safetensors not installed")


class TestExportFromCheckpoint:
    """Test export from checkpoint files."""

    @pytest.mark.slow
    def test_export_creates_parent_dir(self, small_model: FasterRCNN, temp_dir: Path) -> None:
        """Test that export creates parent directories."""
        output_path = temp_dir / "nested" / "dir" / "model.safetensors"

        result = export_to_safetensors(
            model=small_model,
            output_path=output_path,
        )

        assert result.exists()
        assert result.parent.exists()
