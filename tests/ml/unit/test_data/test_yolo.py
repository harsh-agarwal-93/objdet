"""Unit tests for YOLO dataset format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

from objdet.core.constants import ClassIndexMode
from objdet.core.exceptions import DataFormatError
from objdet.data.formats.yolo import YOLODataset

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def yolo_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary YOLO-format dataset directory."""
    # Create directories
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create test images
    img1 = Image.new("RGB", (640, 480), color="red")
    img1.save(images_dir / "image_001.jpg")

    img2 = Image.new("RGB", (800, 600), color="blue")
    img2.save(images_dir / "image_002.jpg")

    img3 = Image.new("RGB", (320, 240), color="green")
    img3.save(images_dir / "image_003.png")

    # Create YOLO label files
    # Format: class_id x_center y_center width height (normalized)
    # Image 001: person at center (0.5, 0.5) with 50% width/height
    label1_content = """0 0.5 0.5 0.5 0.5
1 0.25 0.25 0.2 0.3
"""
    (labels_dir / "image_001.txt").write_text(label1_content)

    # Image 002: single object
    label2_content = "2 0.7 0.8 0.4 0.3\n"
    (labels_dir / "image_002.txt").write_text(label2_content)

    # Image 003: no label file (empty annotations)

    yield tmp_path


class TestYOLODataset:
    """Tests for YOLODataset."""

    def test_init_finds_images(self, yolo_data_dir: Path) -> None:
        """Test that initialization finds all images."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        assert len(dataset) == 3

    def test_len(self, yolo_data_dir: Path) -> None:
        """Test __len__ returns correct count."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        assert len(dataset) == 3

    def test_getitem_returns_correct_format(self, yolo_data_dir: Path) -> None:
        """Test __getitem__ returns tuple of (tensor, target)."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        image, target = dataset[0]

        # Check image is a tensor with correct shape
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.shape[0] == 3  # RGB channels

        # Check target has required keys
        assert "boxes" in target
        assert "labels" in target
        assert "area" in target
        assert "iscrowd" in target
        assert "image_id" in target

    def test_parse_label_file(self, yolo_data_dir: Path) -> None:
        """Test parsing YOLO label files."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        # Find image_001 which has 2 objects
        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_001" in str(img_path):
                _, target = dataset[idx]
                assert target["boxes"].shape[0] == 2
                assert len(target["labels"]) == 2
                break

    def test_parse_label_file_empty(self, yolo_data_dir: Path) -> None:
        """Test handling of missing label files."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        # Find image_003 which has no label file
        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_003" in str(img_path):
                _, target = dataset[idx]
                assert target["boxes"].shape == (0, 4)
                assert len(target["labels"]) == 0
                break

    def test_coordinate_conversion(self, yolo_data_dir: Path) -> None:
        """Test YOLO normalized coords to xyxy conversion."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        # Find image_001 and check first box conversion
        # Original: 0.5 0.5 0.5 0.5 on 640x480 image
        # x_center = 0.5 * 640 = 320, y_center = 0.5 * 480 = 240
        # width = 0.5 * 640 = 320, height = 0.5 * 480 = 240
        # x1 = 320 - 160 = 160, y1 = 240 - 120 = 120
        # x2 = 320 + 160 = 480, y2 = 240 + 120 = 360
        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_001" in str(img_path):
                _, target = dataset[idx]
                expected_box = torch.tensor([160.0, 120.0, 480.0, 360.0])
                assert torch.allclose(target["boxes"][0], expected_box, atol=1.0)
                break

    def test_class_index_mode_yolo(self, yolo_data_dir: Path) -> None:
        """Test YOLO mode keeps labels 0-indexed."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
            class_index_mode=ClassIndexMode.YOLO,
        )

        # Find image_001 with class 0 and 1
        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_001" in str(img_path):
                _, target = dataset[idx]
                # Labels should match file values (0, 1)
                assert 0 in target["labels"].tolist()
                break

    def test_class_index_mode_torchvision(self, yolo_data_dir: Path) -> None:
        """Test TORCHVISION mode adds 1 to labels."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
            class_index_mode=ClassIndexMode.TORCHVISION,
        )

        # Find image_001 - labels should be shifted by 1
        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_001" in str(img_path):
                _, target = dataset[idx]
                # Original class 0 should now be 1
                assert target["labels"].min().item() >= 1
                break

    def test_raises_error_for_empty_directory(self, tmp_path: Path) -> None:
        """Test that empty image directory raises DataFormatError."""
        images_dir = tmp_path / "empty_images"
        labels_dir = tmp_path / "empty_labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        with pytest.raises(DataFormatError):
            YOLODataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                class_names=["person", "car"],
            )

    def test_multiple_image_formats(self, yolo_data_dir: Path) -> None:
        """Test that dataset finds jpg, png, and other formats."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        # Should find .jpg and .png files
        extensions = {p.suffix.lower() for p in dataset.image_paths}
        assert ".jpg" in extensions
        assert ".png" in extensions

    def test_area_calculation(self, yolo_data_dir: Path) -> None:
        """Test that box areas are calculated correctly."""
        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car", "dog"],
        )

        # Find image_001 first box: 160,120,480,360
        # Area = (480-160) * (360-120) = 320 * 240 = 76800
        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_001" in str(img_path):
                _, target = dataset[idx]
                expected_area = (480 - 160) * (360 - 120)
                assert abs(target["area"][0].item() - expected_area) < 100  # Allow small tolerance
                break

    def test_malformed_label_line(self, yolo_data_dir: Path) -> None:
        """Test that malformed lines in label file are skipped."""
        # Create a label file with a bad line
        bad_content = "0 0.5 0.5 0.5 0.5\n0 0.2\n1 0.25 0.25 0.2 0.3"
        (yolo_data_dir / "labels" / "image_001.txt").write_text(bad_content)

        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car"],
        )

        for idx in range(len(dataset)):
            img_path = dataset.image_paths[idx]
            if "image_001" in str(img_path):
                _, target = dataset[idx]
                # Should find 2 valid boxes, skipping the middle line
                assert target["boxes"].shape[0] == 2
                break

    def test_transforms_applied(self, yolo_data_dir: Path) -> None:
        """Test that transforms are applied to image and target."""

        # Mock transform
        from unittest.mock import MagicMock

        mock_transform = MagicMock()
        mock_transform.return_value = (torch.zeros(3, 10, 10), {"boxes": torch.zeros(0, 4)})

        dataset = YOLODataset(
            images_dir=yolo_data_dir / "images",
            labels_dir=yolo_data_dir / "labels",
            class_names=["person", "car"],
            transforms=mock_transform,
        )

        _ = dataset[0]

        assert mock_transform.called
