"""Unit tests for the Predictor class."""

import math
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from torch import Tensor

from objdet.core.exceptions import InferenceError, ModelError
from objdet.core.types import DetectionPrediction
from objdet.inference.predictor import Predictor


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock detection model."""
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    # Mock forward pass
    def forward_side_effect(images: list[Tensor]) -> list[dict[str, Tensor]]:
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
                "labels": torch.tensor([1, 2]),
                "scores": torch.tensor([0.9, 0.1]),
            }
            for _ in images
        ]

    model.side_effect = forward_side_effect
    return model


@pytest.fixture
def predictor(mock_model: MagicMock) -> Predictor:
    """Create a Predictor instance."""
    return Predictor(
        model=mock_model,
        device="cpu",
        confidence_threshold=0.5,
    )


def test_init(mock_model: MagicMock) -> None:
    """Test predictor initialization."""
    predictor = Predictor(mock_model, device="cpu")
    assert predictor.model == mock_model
    assert predictor.device == torch.device("cpu")
    assert mock_model.to.called
    assert mock_model.eval.called


@patch("torch.load")
def test_from_checkpoint_success(mock_torch_load: MagicMock, tmp_path: Path) -> None:
    """Test loading predictor from checkpoint."""
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.touch()

    mock_torch_load.return_value = {"hyper_parameters": {"model_type": "mock_model"}}

    mock_model_class = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_class.load_from_checkpoint.return_value = mock_model_instance

    with patch("objdet.models.MODEL_REGISTRY") as mock_registry:
        mock_registry.__contains__.return_value = True
        mock_registry.get.return_value = mock_model_class

        predictor = Predictor.from_checkpoint(ckpt_path, device="cpu")
        assert isinstance(predictor, Predictor)
        assert predictor.model == mock_model_instance


def test_from_checkpoint_not_found() -> None:
    """Test from_checkpoint with missing file."""
    with pytest.raises(ModelError, match="Checkpoint not found"):
        Predictor.from_checkpoint("nonexistent.ckpt")


def test_predict_tensor(predictor: Predictor) -> None:
    """Test prediction with tensor input."""
    img = torch.randn(3, 100, 100)
    res = predictor.predict(img)
    assert isinstance(res, dict)
    assert "boxes" in res
    assert len(res["boxes"]) == 1  # One above 0.5 threshold
    assert math.isclose(res["scores"][0].item(), 0.9, rel_tol=1e-5)


def test_predict_pil(predictor: Predictor) -> None:
    """Test prediction with PIL image input."""
    img = Image.new("RGB", (100, 100))
    res = predictor.predict(img)
    assert isinstance(res, dict)
    assert len(res["boxes"]) == 1


@patch("PIL.Image.open")
def test_predict_path(mock_open: MagicMock, predictor: Predictor, tmp_path: Path) -> None:
    """Test prediction with image path."""
    img_path = tmp_path / "test.jpg"
    img_path.touch()

    mock_img = Image.new("RGB", (100, 100))
    mock_open.return_value = mock_img

    res = predictor.predict(img_path)
    assert isinstance(res, dict)
    assert len(res["boxes"]) == 1


def test_predict_return_image(predictor: Predictor) -> None:
    """Test prediction returning image tensor."""
    img = torch.randn(3, 100, 100)
    res, img_tensor = predictor.predict(img, return_image=True)
    assert isinstance(res, dict)
    assert isinstance(img_tensor, Tensor)


def test_predict_batch(predictor: Predictor) -> None:
    """Test batch prediction."""
    imgs = [torch.randn(3, 100, 100) for _ in range(5)]
    results = predictor.predict_batch(cast("list[str | Path | Tensor]", imgs), batch_size=2)
    assert len(results) == 5
    assert len(results[0]["boxes"]) == 1


def test_predict_directory(predictor: Predictor, tmp_path: Path) -> None:
    """Test prediction on directory of images."""
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.png").touch()
    (tmp_path / "other.txt").touch()

    with patch(
        "objdet.inference.predictor.Predictor._load_image", return_value=torch.randn(3, 10, 10)
    ):
        results = predictor.predict_directory(tmp_path)
        assert len(results) == 2
        assert "img1.jpg" in results
        assert "img2.png" in results


def test_predict_directory_error(predictor: Predictor, tmp_path: Path) -> None:
    """Test predict_directory with non-existent directory."""
    with pytest.raises(InferenceError, match="Not a directory"):
        predictor.predict_directory(tmp_path / "ghost")


def test_postprocess(predictor: Predictor) -> None:
    """Test post-processing thresholding."""
    pred = cast(
        "DetectionPrediction",
        {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.6, 0.4]),
        },
    )
    # Predictor has threshold 0.5
    processed = predictor._postprocess(pred)
    assert len(processed["boxes"]) == 1
    assert math.isclose(processed["scores"][0].item(), 0.6, rel_tol=1e-5)
