"""Shared fixtures for integration tests."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_dir(temp_dir: Path) -> Path:
    """Create a directory with sample images for testing."""
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True)

    # Create sample images
    for i in range(3):
        img = Image.new("RGB", (640, 480), color=(i * 50, 100, 150))
        img.save(images_dir / f"image_{i:03d}.jpg")

    return images_dir


@pytest.fixture
def sample_checkpoint(temp_dir: Path) -> Path:
    """Create a minimal checkpoint file for testing.

    Note: This creates a dummy file, not a real model checkpoint.
    For actual model loading tests, use a real trained checkpoint.
    """
    checkpoint_path = temp_dir / "test_model.ckpt"

    # Create a minimal state dict structure
    state = {
        "state_dict": {},
        "hyper_parameters": {
            "num_classes": 5,
            "class_index_mode": "torchvision",
        },
    }
    torch.save(state, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def voc_dataset_dir(temp_dir: Path) -> Path:
    """Create a minimal VOC-format dataset for integration testing."""
    # Create directories
    images_dir = temp_dir / "JPEGImages"
    annotations_dir = temp_dir / "Annotations"
    splits_dir = temp_dir / "ImageSets" / "Main"

    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    # Create images
    for i in range(2):
        img = Image.new("RGB", (640, 480), color=(i * 100, 50, 100))
        img.save(images_dir / f"img_{i:04d}.jpg")

        # Create annotation
        xml = f"""<?xml version="1.0"?>
<annotation>
    <filename>img_{i:04d}.jpg</filename>
    <size><width>640</width><height>480</height><depth>3</depth></size>
    <object>
        <name>person</name>
        <bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>
    </object>
</annotation>"""
        (annotations_dir / f"img_{i:04d}.xml").write_text(xml)

    # Create split file
    (splits_dir / "train.txt").write_text("img_0000\n")
    (splits_dir / "val.txt").write_text("img_0001\n")

    return temp_dir


@pytest.fixture
def trained_checkpoint(temp_dir: Path) -> Path:
    """Create a minimal trained checkpoint with actual model weights.

    This creates a checkpoint that can be loaded for export tests.
    Uses a simple model structure compatible with the export functions.
    """
    from torch import nn

    # Create a simple model with actual weights
    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 640 * 640, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            return x

    model = SimpleModel()
    checkpoint_path = temp_dir / "trained_model.ckpt"

    state = {
        "state_dict": model.state_dict(),
        "hyper_parameters": {
            "num_classes": 5,
            "class_index_mode": "torchvision",
        },
        "epoch": 1,
        "global_step": 100,
    }
    torch.save(state, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def sample_prediction() -> dict[str, torch.Tensor]:
    """Create a sample detection prediction dictionary."""
    return {
        "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0], [150.0, 150.0, 300.0, 300.0]]),
        "labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.95, 0.87]),
    }


@pytest.fixture
def sample_target() -> dict[str, torch.Tensor]:
    """Create a sample detection target dictionary."""
    boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0], [150.0, 150.0, 300.0, 300.0]])
    return {
        "boxes": boxes,
        "labels": torch.tensor([1, 2]),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "iscrowd": torch.zeros(2, dtype=torch.int64),
    }
