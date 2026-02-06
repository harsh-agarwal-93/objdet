"""Unit tests for inference predictor."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from objdet.inference.predictor import Predictor


class MockModel:
    """Mock detection model for testing."""

    def __init__(self):
        self.training = True

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, images):
        """Return mock predictions."""
        predictions = []
        for img in images:
            predictions.append(
                {
                    "boxes": torch.tensor([[100, 100, 200, 200]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
            )
        return predictions


@pytest.fixture
def mock_model():
    """Create mock model."""
    return MockModel()


@pytest.fixture
def predictor(mock_model):
    """Create predictor with mock model."""
    return Predictor(
        model=mock_model,
        device="cpu",
        confidence_threshold=0.25,
    )


class TestPredictor:
    """Tests for Predictor class."""

    def test_init_sets_eval_mode(self, predictor, mock_model):
        """Predictor should set model to eval mode."""
        assert not mock_model.training

    def test_predict_single_image(self, predictor):
        """Test prediction on single image tensor."""
        image = torch.rand(3, 480, 640)
        result = predictor.predict(image)

        assert "boxes" in result
        assert "labels" in result
        assert "scores" in result

    def test_predict_filters_by_confidence(self):
        """Test that low confidence predictions are filtered."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.return_value = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.1]),  # Below threshold
            }
        ]

        predictor = Predictor(
            model=mock_model,
            device="cpu",
            confidence_threshold=0.5,
        )

        image = torch.rand(3, 480, 640)
        result = predictor.predict(image)

        from typing import cast

        result = cast("dict", result)
        assert len(result["boxes"]) == 0

    def test_predict_batch(self, predictor):
        """Test batch prediction."""
        images = [torch.rand(3, 480, 640) for _ in range(3)]
        results = predictor.predict_batch(images, batch_size=2)

        assert len(results) == 3
        for result in results:
            assert "boxes" in result


class TestPredictorLoadImage:
    """Tests for image loading in Predictor."""

    def test_load_tensor(self, predictor):
        """Tensor input should pass through."""
        image = torch.rand(3, 480, 640)
        result = predictor._load_image(image)

        assert torch.equal(result, image)

    @patch("PIL.Image.open")
    def test_load_path(self, mock_open, predictor):
        """Path input should load image."""
        import numpy as np

        # Mock PIL image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        # Fix: Accept any arguments for __array__ as np.array calls it with args
        mock_img.__array__ = lambda *args, **kwargs: (np.random.rand(480, 640, 3) * 255).astype(
            np.uint8
        )
        mock_open.return_value = mock_img

        result = predictor._load_image("/path/to/image.jpg")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # Channels first
