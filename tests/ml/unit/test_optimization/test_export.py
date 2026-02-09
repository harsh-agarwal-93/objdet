"""Unit tests for model export utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from objdet.core.exceptions import ExportError
from objdet.optimization.export import (
    _load_model_from_checkpoint,
    export_model,
    export_to_onnx,
    export_to_safetensors,
    export_to_tensorrt,
)


@pytest.fixture
def mock_detector() -> MagicMock:
    """Create a mock detector model."""
    model = MagicMock()
    model.num_classes = 3
    model.class_index_mode.value = "torchvision"
    model.state_dict.return_value = {"weight": torch.tensor([1.0])}
    model.eval.return_value = model
    return model


@patch("objdet.optimization.export._load_model_from_checkpoint")
@patch("torch.onnx.export")
def test_export_to_onnx(
    mock_onnx_export: MagicMock, mock_load: MagicMock, mock_detector: MagicMock, tmp_path: Path
) -> None:
    """Test ONNX export logic."""
    mock_load.return_value = mock_detector
    output_path = tmp_path / "exp/model.onnx"

    # Test with model instance
    res = export_to_onnx(mock_detector, output_path)
    assert res == output_path
    assert mock_onnx_export.called
    assert output_path.parent.exists()

    # Test with path (triggers _load_model_from_checkpoint)
    mock_onnx_export.reset_mock()
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.touch()
    export_to_onnx(ckpt_path, output_path)
    assert mock_load.called
    assert mock_onnx_export.called


@patch("safetensors.torch.save_file")
def test_export_to_safetensors(
    mock_save_file: MagicMock, mock_detector: MagicMock, tmp_path: Path
) -> None:
    """Test SafeTensors export logic."""
    output_path = tmp_path / "model.safetensors"

    res = export_to_safetensors(mock_detector, output_path)
    assert res == output_path
    assert mock_save_file.called

    # Verify metadata
    _, kwargs = mock_save_file.call_args
    assert "metadata" in kwargs
    assert kwargs["metadata"]["num_classes"] == "3"


@patch("objdet.optimization.export.export_to_onnx")
@patch("objdet.optimization.export.export_to_safetensors")
@patch("objdet.optimization.export.export_to_tensorrt")
@patch("objdet.optimization.export._load_model_from_checkpoint")
def test_export_model_main_entry(
    mock_load: MagicMock,
    mock_to_trt: MagicMock,
    mock_to_safe: MagicMock,
    mock_to_onnx: MagicMock,
    mock_detector: MagicMock,
    tmp_path: Path,
) -> None:
    """Test the main export_model entry point for all formats."""
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.touch()
    output_path = tmp_path / "model.onnx"
    mock_load.return_value = mock_detector

    # ONNX
    export_model(ckpt_path, output_path, export_format="onnx")
    assert mock_to_onnx.called

    # SafeTensors
    export_model(ckpt_path, output_path, export_format="safetensors")
    assert mock_to_safe.called

    # TensorRT
    export_model(ckpt_path, output_path, export_format="tensorrt")
    assert mock_to_trt.called


def test_export_model_missing_checkpoint() -> None:
    """Test error when checkpoint is missing."""
    with pytest.raises(ExportError, match="Checkpoint not found"):
        export_model("nonexistent.ckpt", "out.onnx")


@patch("objdet.optimization.export._load_model_from_checkpoint")
def test_export_model_invalid_format(mock_load: MagicMock, tmp_path: Path) -> None:
    """Test error when format is unknown."""
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.touch()
    with pytest.raises(ExportError, match="Unknown export format"):
        export_model(ckpt_path, "out.bin", export_format="invalid")  # type: ignore


@patch("torch.load")
def test_load_model_from_checkpoint_success(mock_torch_load: MagicMock, tmp_path: Path) -> None:
    """Test successful model loading from checkpoint metadata."""
    mock_torch_load.return_value = {"hyper_parameters": {"model_type": "mock_model"}}
    mock_model_class = MagicMock()

    with patch("objdet.models.MODEL_REGISTRY") as mock_registry:
        mock_registry.__contains__.return_value = True
        mock_registry.get.return_value = mock_model_class

        ckpt_path = tmp_path / "model.ckpt"
        ckpt_path.touch()

        _load_model_from_checkpoint(ckpt_path)
        assert mock_registry.get.called
        assert mock_model_class.load_from_checkpoint.called


@patch("torch.load")
def test_load_model_from_checkpoint_fail(mock_torch_load: MagicMock, tmp_path: Path) -> None:
    """Test failure in _load_model_from_checkpoint when metadata is missing."""
    mock_torch_load.return_value = {"weights": {}}
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.touch()

    with pytest.raises(ExportError, match="Cannot determine model class"):
        _load_model_from_checkpoint(ckpt_path)


@patch("objdet.optimization.export.get_logger")
@patch("torch.onnx.export")
def test_export_to_onnx_simplify(
    mock_onnx_export: MagicMock,
    mock_get_logger: MagicMock,
    mock_detector: MagicMock,
    tmp_path: Path,
) -> None:
    """Test ONNX export with simplification."""
    output_path = tmp_path / "model.onnx"

    # Mock onnx and onnxsim
    mock_onnx = MagicMock()
    mock_onnxsim = MagicMock()
    mock_onnxsim.simplify.return_value = (MagicMock(), True)

    with patch.dict("sys.modules", {"onnx": mock_onnx, "onnxsim": mock_onnxsim}):
        export_to_onnx(mock_detector, output_path, simplify=True)
        assert mock_onnx.load.called
        assert mock_onnxsim.simplify.called
        assert mock_onnx.save.called


@patch("objdet.optimization.export.get_logger")
@patch("torch.jit.save")
def test_export_to_tensorrt_success(
    mock_jit_save: MagicMock, mock_get_logger: MagicMock, mock_detector: MagicMock, tmp_path: Path
) -> None:
    """Test TensorRT export success (mocked)."""
    output_path = tmp_path / "model.trt"
    mock_detector.cuda.return_value = mock_detector

    mock_trt = MagicMock()
    mock_trt.Input = MagicMock
    mock_trt.compile.return_value = MagicMock()

    with patch.dict("sys.modules", {"torch_tensorrt": mock_trt}):
        res = export_to_tensorrt(mock_detector, output_path)
        assert res == output_path
        assert mock_trt.compile.called
        assert mock_jit_save.called


def test_export_to_safetensors_import_error(mock_detector: MagicMock, tmp_path: Path) -> None:
    """Test SafeTensors export with missing dependency."""
    # We can mock the import by making it fail
    with (
        patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                raise_importerror() if "safetensors" in name else __import__(name, *args, **kwargs)
            ),
        ),
        pytest.raises(ExportError, match="safetensors is required"),
    ):
        export_to_safetensors(mock_detector, tmp_path / "out.safetensors")


def test_export_to_tensorrt_import_error(mock_detector: MagicMock, tmp_path: Path) -> None:
    """Test TensorRT export with missing dependency."""
    with (
        patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                raise_importerror()
                if "torch_tensorrt" in name
                else __import__(name, *args, **kwargs)
            ),
        ),
        pytest.raises(ExportError, match="torch-tensorrt is required"),
    ):
        export_to_tensorrt(mock_detector, tmp_path / "out.trt")


def raise_importerror():
    """Helper to raise ImportError."""
    raise ImportError("Mocked ImportError")
