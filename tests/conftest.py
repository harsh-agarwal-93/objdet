"""Pytest configuration and shared fixtures for ObjDet tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Return temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Return temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Return CPU device explicitly."""
    return torch.device("cpu")


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def sample_image() -> Tensor:
    """Return a sample image tensor (3, 640, 640)."""
    return torch.rand(3, 640, 640)


@pytest.fixture
def sample_image_batch() -> Tensor:
    """Return a batch of sample images (4, 3, 640, 640)."""
    return torch.rand(4, 3, 640, 640)


@pytest.fixture
def sample_boxes() -> Tensor:
    """Return sample bounding boxes (N, 4) in xyxy format."""
    return torch.tensor(
        [
            [100.0, 100.0, 200.0, 200.0],
            [150.0, 150.0, 300.0, 300.0],
            [50.0, 50.0, 150.0, 150.0],
        ]
    )


@pytest.fixture
def sample_labels() -> Tensor:
    """Return sample class labels (N,)."""
    return torch.tensor([1, 2, 1])


@pytest.fixture
def sample_scores() -> Tensor:
    """Return sample confidence scores (N,)."""
    return torch.tensor([0.95, 0.87, 0.72])


@pytest.fixture
def sample_target(sample_boxes: Tensor, sample_labels: Tensor) -> dict[str, Tensor]:
    """Return a sample detection target dictionary."""
    return {
        "boxes": sample_boxes,
        "labels": sample_labels,
        "area": (sample_boxes[:, 2] - sample_boxes[:, 0])
        * (sample_boxes[:, 3] - sample_boxes[:, 1]),
        "iscrowd": torch.zeros(len(sample_boxes), dtype=torch.int64),
    }


@pytest.fixture
def sample_prediction(
    sample_boxes: Tensor,
    sample_labels: Tensor,
    sample_scores: Tensor,
) -> dict[str, Tensor]:
    """Return a sample detection prediction dictionary."""
    return {
        "boxes": sample_boxes,
        "labels": sample_labels,
        "scores": sample_scores,
    }


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def num_classes() -> int:
    """Return default number of classes for testing."""
    return 10


@pytest.fixture
def class_names(num_classes: int) -> list[str]:
    """Return class names for testing."""
    return [f"class_{i}" for i in range(num_classes)]


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Return base configuration dictionary for testing."""
    return {
        "model": {
            "num_classes": 10,
            "pretrained": False,
        },
        "data": {
            "batch_size": 2,
            "num_workers": 0,
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "fast_dev_run": True,
        },
    }


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_cuda() -> Generator[None, None, None]:
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify test collection based on markers and available hardware."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
