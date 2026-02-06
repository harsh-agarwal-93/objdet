"""Integration tests for serving functionality.

These tests verify that the serving API and server correctly handle
requests and produce valid detection responses.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

# =============================================================================
# DetectionAPI Tests
# =============================================================================


class TestDetectionAPI:
    """Test DetectionAPI class."""

    def test_api_initialization(self, temp_dir: Path) -> None:
        """Test API initializes with valid parameters."""
        from objdet.serving.api import DetectionAPI

        # Create a dummy checkpoint
        checkpoint_path = temp_dir / "model.ckpt"
        state = {
            "state_dict": {},
            "hyper_parameters": {"num_classes": 5},
        }
        torch.save(state, checkpoint_path)

        api = DetectionAPI(
            checkpoint_path=checkpoint_path,
            device="cpu",
            confidence_threshold=0.25,
            max_batch_size=8,
        )

        assert api.checkpoint_path == checkpoint_path
        assert api.device == "cpu"
        assert api.confidence_threshold == 0.25

    def test_decode_request_base64_image(self, temp_dir: Path) -> None:
        """Test decoding base64-encoded image from request."""
        from objdet.serving.api import DetectionAPI

        # Create a dummy API (won't call setup)
        checkpoint_path = temp_dir / "model.ckpt"
        torch.save({"state_dict": {}, "hyper_parameters": {"num_classes": 5}}, checkpoint_path)

        api = DetectionAPI(checkpoint_path=checkpoint_path, device="cpu")

        # Create a test image and encode as base64
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Mock the model to avoid setup
        api.model = MagicMock()

        request = {"image": img_b64}
        result = api.decode_request(request)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 3  # (C, H, W)
        assert result.shape[0] == 3  # RGB

    def test_decode_request_tensor(self, temp_dir: Path) -> None:
        """Test decoding raw tensor data from request."""
        from objdet.serving.api import DetectionAPI

        checkpoint_path = temp_dir / "model.ckpt"
        torch.save({"state_dict": {}, "hyper_parameters": {"num_classes": 5}}, checkpoint_path)

        api = DetectionAPI(checkpoint_path=checkpoint_path, device="cpu")
        api.model = MagicMock()

        # Tensor as list
        tensor_data = torch.rand(3, 224, 224).tolist()
        request = {"tensor": tensor_data}

        result = api.decode_request(request)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_encode_response_format(self, temp_dir: Path) -> None:
        """Test that response encoding produces JSON-serializable output."""
        from objdet.serving.api import DetectionAPI

        checkpoint_path = temp_dir / "model.ckpt"
        torch.save({"state_dict": {}, "hyper_parameters": {"num_classes": 5}}, checkpoint_path)

        api = DetectionAPI(checkpoint_path=checkpoint_path, device="cpu")

        output = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 80.0, 80.0]]),
            "labels": torch.tensor([1, 2]),
            "scores": torch.tensor([0.95, 0.87]),
        }

        response = api.encode_response(output)

        assert isinstance(response, dict)
        # API returns 'detections' format, not raw boxes
        assert "detections" in response or "boxes" in response

        # Should be JSON-serializable (lists, not tensors)
        import json

        json.dumps(response)  # Should not raise


# =============================================================================
# ABTestingAPI Tests
# =============================================================================


class TestABTestingAPI:
    """Test ABTestingAPI class."""

    def test_ab_testing_initialization(self, temp_dir: Path) -> None:
        """Test A/B testing API initializes correctly."""
        from objdet.serving.api import ABTestingAPI

        # Create dummy checkpoints
        checkpoint_a = temp_dir / "model_a.ckpt"
        checkpoint_b = temp_dir / "model_b.ckpt"
        torch.save({"state_dict": {}, "hyper_parameters": {"num_classes": 5}}, checkpoint_a)
        torch.save({"state_dict": {}, "hyper_parameters": {"num_classes": 5}}, checkpoint_b)

        models: dict[str, tuple[str, float]] = {
            "model_a": (str(checkpoint_a), 0.7),
            "model_b": (str(checkpoint_b), 0.3),
        }

        api = ABTestingAPI(models=models, device="cpu")  # type: ignore[arg-type]

        assert len(api.model_configs) == 2
        assert "model_a" in api.model_configs
        assert "model_b" in api.model_configs

    @pytest.mark.skip(reason="Requires setup() which loads actual models")
    def test_ab_testing_model_selection_weights(self, temp_dir: Path) -> None:
        """Test that model selection respects traffic weights."""
        pass


# =============================================================================
# Server Integration Tests
# =============================================================================


class TestServerIntegration:
    """Test server startup and configuration."""

    def test_run_server_requires_checkpoint(self, temp_dir: Path) -> None:
        """Test that run_server raises error without checkpoint."""
        from objdet.serving.server import run_server

        # Empty config with no checkpoint
        config_path = temp_dir / "config.yaml"
        config_path.write_text("dummy_key: value\n")

        with pytest.raises((ValueError, AttributeError, TypeError)):  # type: ignore[call-overload]
            run_server(config_path=config_path)

    def test_load_config_reads_yaml(self, temp_dir: Path) -> None:
        """Test that config loading works correctly."""
        from objdet.serving.server import _load_config

        config_path = temp_dir / "config.yaml"
        config_path.write_text(
            """
checkpoint_path: /path/to/model.ckpt
confidence_threshold: 0.3
host: localhost
port: 9000
"""
        )

        config = _load_config(config_path)

        assert config["checkpoint_path"] == "/path/to/model.ckpt"
        assert config["confidence_threshold"] == 0.3
        assert config["host"] == "localhost"
        assert config["port"] == 9000

    def test_load_config_missing_file(self, temp_dir: Path) -> None:
        """Test that missing config file raises error."""
        from objdet.serving.server import _load_config

        with pytest.raises(FileNotFoundError):
            _load_config(temp_dir / "nonexistent.yaml")


# =============================================================================
# CLI Serve Integration Tests
# =============================================================================


class TestCLIServeIntegration:
    """Test serve command through CLI."""

    def test_cli_serve_help(self) -> None:
        """Test serve subcommand help."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "objdet", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0
        assert "serve" in result.stdout.lower() or "config" in result.stdout.lower()

    def test_cli_serve_missing_config_fails(self, temp_dir: Path) -> None:
        """Test that serve fails with missing config."""
        import subprocess
        import sys

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

        # Should fail with error about missing config
        assert result.returncode != 0 or "error" in result.stderr.lower()
