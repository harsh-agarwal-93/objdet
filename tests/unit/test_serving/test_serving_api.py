"""Unit tests for serving API."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image


class TestDetectionAPI:
    """Tests for DetectionAPI class."""

    @pytest.fixture
    def mock_checkpoint(self, tmp_path: Path) -> Path:
        """Create a mock checkpoint file."""
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.touch()
        return checkpoint

    def test_init(self, mock_checkpoint: Path) -> None:
        """Test initialization with parameters."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(
            checkpoint_path=mock_checkpoint,
            device="cpu",
            confidence_threshold=0.5,
            max_batch_size=4,
        )

        assert api.checkpoint_path == mock_checkpoint
        assert api.device == "cpu"
        assert api.confidence_threshold == 0.5
        assert api.max_batch_size == 4
        assert api.model is None  # Not loaded until setup()

    def test_init_defaults(self, mock_checkpoint: Path) -> None:
        """Test initialization with default parameters."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)

        assert api.device == "cuda"
        assert api.confidence_threshold == 0.25
        assert api.max_batch_size == 8

    def test_decode_request_base64(self, mock_checkpoint: Path) -> None:
        """Test decoding base64 encoded image from request."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)

        # Create a test image and encode it
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        request = {"image": img_base64}
        tensor = api.decode_request(request)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 3  # RGB
        assert tensor.shape[1] == 100
        assert tensor.shape[2] == 100

    def test_decode_request_tensor(self, mock_checkpoint: Path) -> None:
        """Test decoding raw tensor from request."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)

        tensor_data = torch.randn(3, 64, 64).tolist()
        request = {"tensor": tensor_data}

        tensor = api.decode_request(request)

        assert isinstance(tensor, torch.Tensor)

    def test_decode_request_missing_field_raises(self, mock_checkpoint: Path) -> None:
        """Test that missing required field raises error."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)

        request = {"invalid_field": "data"}

        with pytest.raises(ValueError, match="must contain"):
            api.decode_request(request)

    def test_encode_response_basic(self, mock_checkpoint: Path) -> None:
        """Test encoding prediction to response."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)

        output = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 150.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.95]),
        }

        response = api.encode_response(output)

        assert "detections" in response
        assert "count" in response
        assert response["count"] == 1
        assert len(response["detections"]) == 1

        detection = response["detections"][0]
        assert "box" in detection
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert detection["box"]["x1"] == 10.0
        assert abs(detection["confidence"] - 0.95) < 0.01

    def test_encode_response_with_class_names(self, mock_checkpoint: Path) -> None:
        """Test encoding with custom class names."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)
        api.class_names = ["person", "car", "dog"]

        output = {
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 150.0]]),
            "labels": torch.tensor([1]),  # Should map to "car"
            "scores": torch.tensor([0.9]),
        }

        response = api.encode_response(output)

        assert response["detections"][0]["class_name"] == "car"

    def test_encode_response_empty(self, mock_checkpoint: Path) -> None:
        """Test encoding empty prediction."""
        from objdet.serving.api import DetectionAPI

        api = DetectionAPI(checkpoint_path=mock_checkpoint)

        output = {
            "boxes": torch.empty(0, 4),
            "labels": torch.empty(0, dtype=torch.int64),
            "scores": torch.empty(0),
        }

        response = api.encode_response(output)

        assert response["count"] == 0
        assert response["detections"] == []


class TestABTestingAPI:
    """Tests for ABTestingAPI class."""

    def test_init(self) -> None:
        """Test initialization with model configs."""
        from objdet.serving.api import ABTestingAPI

        models: dict[str, tuple[Path | str, float]] = {
            "v1": (Path("/path/model_v1.ckpt"), 0.7),
            "v2": (Path("/path/model_v2.ckpt"), 0.3),
        }

        api = ABTestingAPI(models=models)

        assert len(api.model_configs) == 2
        # Weights should be normalized
        assert abs(sum(api.weights.values()) - 1.0) < 1e-6

    def test_weight_normalization(self) -> None:
        """Test that weights are normalized correctly."""
        from objdet.serving.api import ABTestingAPI

        models: dict[str, tuple[Path | str, float]] = {
            "v1": (Path("/path/model_v1.ckpt"), 2.0),
            "v2": (Path("/path/model_v2.ckpt"), 3.0),
        }

        api = ABTestingAPI(models=models)

        assert abs(api.weights["v1"] - 0.4) < 1e-6
        assert abs(api.weights["v2"] - 0.6) < 1e-6

    def test_select_model_distribution(self) -> None:
        """Test that model selection follows weight distribution."""
        from objdet.serving.api import ABTestingAPI

        models: dict[str, tuple[Path | str, float]] = {
            "v1": (Path("/path/model_v1.ckpt"), 0.8),
            "v2": (Path("/path/model_v2.ckpt"), 0.2),
        }

        api = ABTestingAPI(models=models)

        # Mock APIs
        api.apis = {
            "v1": MagicMock(),
            "v2": MagicMock(),
        }

        # Run many selections and check distribution
        selections = {"v1": 0, "v2": 0}
        for _ in range(1000):
            name, _ = api._select_model()
            selections[name] += 1

        # v1 should be selected more often (roughly 80%)
        v1_ratio = selections["v1"] / 1000
        assert 0.7 < v1_ratio < 0.9  # Allow some variance
