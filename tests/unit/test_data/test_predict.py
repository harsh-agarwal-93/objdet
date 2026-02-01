"""Unit tests for PredictDataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

from objdet.data.formats.predict import PredictDataset

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def predict_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory with test images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create test images with different extensions
    Image.new("RGB", (640, 480), color="red").save(images_dir / "test_001.jpg")
    Image.new("RGB", (800, 600), color="blue").save(images_dir / "test_002.png")
    Image.new("RGB", (320, 240), color="green").save(images_dir / "test_003.JPEG")

    yield images_dir


@pytest.fixture
def empty_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create an empty temporary directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    yield empty_dir


class TestPredictDataset:
    """Tests for PredictDataset class."""

    def test_init_with_valid_directory(self, predict_data_dir: Path) -> None:
        """Test initialization with a directory containing images."""
        dataset = PredictDataset(image_dir=predict_data_dir)

        assert len(dataset) == 3
        assert dataset.image_dir == predict_data_dir
        assert dataset.transforms is None

    def test_init_with_empty_directory(self, empty_data_dir: Path) -> None:
        """Test initialization with empty directory returns empty dataset."""
        dataset = PredictDataset(image_dir=empty_data_dir)

        assert len(dataset) == 0

    def test_len(self, predict_data_dir: Path) -> None:
        """Test __len__ returns correct image count."""
        dataset = PredictDataset(image_dir=predict_data_dir)

        assert len(dataset) == 3

    def test_getitem_returns_tensor_and_metadata(self, predict_data_dir: Path) -> None:
        """Test __getitem__ returns correct output format."""
        dataset = PredictDataset(image_dir=predict_data_dir)

        image_tensor, metadata = dataset[0]

        # Check image tensor
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.ndim == 3
        assert image_tensor.shape[0] == 3  # RGB channels
        assert image_tensor.dtype == torch.float32
        assert image_tensor.min() >= 0.0
        assert image_tensor.max() <= 1.0

        # Check metadata
        assert "image_id" in metadata
        assert "file_path" in metadata
        assert "original_size" in metadata
        assert metadata["image_id"] == 0

    def test_getitem_with_transforms(self, predict_data_dir: Path) -> None:
        """Test that transforms are applied correctly."""

        def mock_transform(image: torch.Tensor, target: dict) -> tuple:
            # Simple transform that scales image by 0.5
            return image * 0.5, target

        dataset = PredictDataset(image_dir=predict_data_dir, transforms=mock_transform)

        image_tensor, _ = dataset[0]

        # Max should be 0.5 or less after transform
        assert image_tensor.max() <= 0.5

    def test_various_extensions(self, tmp_path: Path) -> None:
        """Test that various image extensions are loaded."""
        images_dir = tmp_path / "multi_ext"
        images_dir.mkdir()

        # Create images with different extensions
        Image.new("RGB", (100, 100), color="red").save(images_dir / "img1.jpg")
        Image.new("RGB", (100, 100), color="blue").save(images_dir / "img2.jpeg")
        Image.new("RGB", (100, 100), color="green").save(images_dir / "img3.png")
        Image.new("RGB", (100, 100), color="yellow").save(images_dir / "img4.bmp")
        Image.new("RGB", (100, 100), color="purple").save(images_dir / "img5.webp")

        dataset = PredictDataset(image_dir=images_dir)

        assert len(dataset) == 5

    def test_custom_extensions(self, predict_data_dir: Path) -> None:
        """Test filtering by custom extensions."""
        # Only look for png files
        dataset = PredictDataset(
            image_dir=predict_data_dir,
            extensions=(".png",),
        )

        assert len(dataset) == 1

    def test_original_size_in_metadata(self, predict_data_dir: Path) -> None:
        """Test that original image size is recorded in metadata."""
        dataset = PredictDataset(image_dir=predict_data_dir)

        # Find the 640x480 image
        for i in range(len(dataset)):
            _, metadata = dataset[i]
            if "test_001" in metadata["file_path"]:
                assert metadata["original_size"] == (640, 480)
                break

    def test_image_paths_are_sorted(self, predict_data_dir: Path) -> None:
        """Test that image paths are sorted consistently."""
        dataset = PredictDataset(image_dir=predict_data_dir)

        # Get all paths
        paths = [str(p) for p in dataset.image_paths]

        # Should be sorted
        assert paths == sorted(paths)
