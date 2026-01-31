"""Unit tests for VOC dataset format."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

from objdet.core.constants import ClassIndexMode
from objdet.core.exceptions import DataFormatError
from objdet.data.formats.voc import VOC_CLASSES, VOCDataset

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def voc_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary VOC-format dataset directory."""
    # Create directories
    images_dir = tmp_path / "JPEGImages"
    annotations_dir = tmp_path / "Annotations"
    splits_dir = tmp_path / "ImageSets" / "Main"

    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    # Create a test image
    img = Image.new("RGB", (640, 480), color="red")
    img.save(images_dir / "test_001.jpg")

    # Create another test image
    img2 = Image.new("RGB", (800, 600), color="blue")
    img2.save(images_dir / "test_002.jpg")

    # Create VOC XML annotation for test_001
    xml_content = """<?xml version="1.0"?>
<annotation>
    <folder>VOC2012</folder>
    <filename>test_001.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
            <ymax>300</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
    <object>
        <name>car</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>200</ymin>
            <xmax>500</xmax>
            <ymax>400</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
</annotation>"""
    (annotations_dir / "test_001.xml").write_text(xml_content)

    # Create split file
    (splits_dir / "trainval.txt").write_text("test_001\ntest_002\n")
    (splits_dir / "train.txt").write_text("test_001\n")
    (splits_dir / "val.txt").write_text("test_002\n")

    yield tmp_path


class TestVOCDataset:
    """Tests for VOCDataset."""

    def test_init_with_defaults(self, voc_data_dir: Path) -> None:
        """Test initialization with default VOC classes."""
        dataset = VOCDataset(data_dir=voc_data_dir, split="trainval")

        assert dataset.class_names == VOC_CLASSES
        assert len(dataset) == 2

    def test_init_with_custom_classes(self, voc_data_dir: Path) -> None:
        """Test initialization with custom class names."""
        custom_classes = ["person", "car"]
        dataset = VOCDataset(
            data_dir=voc_data_dir,
            split="trainval",
            class_names=custom_classes,
        )

        assert dataset.class_names == custom_classes

    def test_len(self, voc_data_dir: Path) -> None:
        """Test __len__ returns correct count."""
        dataset = VOCDataset(data_dir=voc_data_dir, split="train")
        assert len(dataset) == 1

        dataset_full = VOCDataset(data_dir=voc_data_dir, split="trainval")
        assert len(dataset_full) == 2

    def test_getitem_returns_correct_format(self, voc_data_dir: Path) -> None:
        """Test __getitem__ returns tuple of (tensor, target)."""
        dataset = VOCDataset(data_dir=voc_data_dir, split="train")

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

    def test_parse_xml_with_objects(self, voc_data_dir: Path) -> None:
        """Test XML parsing extracts bounding boxes correctly."""
        dataset = VOCDataset(
            data_dir=voc_data_dir,
            split="train",
            class_names=["person", "car"],
        )

        image, target = dataset[0]

        # Should have 2 objects
        assert target["boxes"].shape[0] == 2
        assert len(target["labels"]) == 2

        # Check first box coordinates (person: 100,100,200,300)
        assert torch.allclose(
            target["boxes"][0],
            torch.tensor([100.0, 100.0, 200.0, 300.0]),
        )

    def test_parse_xml_empty(self, voc_data_dir: Path) -> None:
        """Test handling of images with no annotations."""
        dataset = VOCDataset(data_dir=voc_data_dir, split="val")

        # test_002 has no annotation XML
        image, target = dataset[0]

        assert target["boxes"].shape == (0, 4)
        assert len(target["labels"]) == 0

    def test_class_index_mode_torchvision(self, voc_data_dir: Path) -> None:
        """Test TORCHVISION mode starts labels at 1."""
        dataset = VOCDataset(
            data_dir=voc_data_dir,
            split="train",
            class_names=["person", "car"],
            class_index_mode=ClassIndexMode.TORCHVISION,
        )

        _, target = dataset[0]

        # Labels should be 1-indexed (background at 0)
        assert target["labels"].min().item() >= 1

    def test_class_index_mode_yolo(self, voc_data_dir: Path) -> None:
        """Test YOLO mode starts labels at 0."""
        dataset = VOCDataset(
            data_dir=voc_data_dir,
            split="train",
            class_names=["person", "car"],
            class_index_mode=ClassIndexMode.YOLO,
        )

        _, target = dataset[0]

        # Labels should be 0-indexed
        assert target["labels"].min().item() >= 0

    def test_missing_split_file_fallback(self, tmp_path: Path) -> None:
        """Test fallback to globbing when split file is missing."""
        # Create minimal directory without split files
        images_dir = tmp_path / "JPEGImages"
        annotations_dir = tmp_path / "Annotations"
        images_dir.mkdir(parents=True)
        annotations_dir.mkdir(parents=True)

        # Create image
        img = Image.new("RGB", (100, 100), color="green")
        img.save(images_dir / "fallback_image.jpg")

        # This should still work by globbing images
        dataset = VOCDataset(data_dir=tmp_path, split="nonexistent")

        assert len(dataset) == 1

    def test_raises_error_for_empty_dataset(self, tmp_path: Path) -> None:
        """Test that empty dataset raises DataFormatError."""
        # Create empty directories
        (tmp_path / "JPEGImages").mkdir(parents=True)
        (tmp_path / "Annotations").mkdir(parents=True)
        (tmp_path / "ImageSets" / "Main").mkdir(parents=True)
        (tmp_path / "ImageSets" / "Main" / "train.txt").write_text("")

        with pytest.raises(DataFormatError):
            VOCDataset(data_dir=tmp_path, split="train")

    def test_area_calculation(self, voc_data_dir: Path) -> None:
        """Test that box areas are calculated correctly."""
        dataset = VOCDataset(
            data_dir=voc_data_dir,
            split="train",
            class_names=["person", "car"],
        )

        _, target = dataset[0]

        # First box: (200-100) * (300-100) = 100 * 200 = 20000
        expected_area_1 = (200 - 100) * (300 - 100)
        assert target["area"][0].item() == expected_area_1

        # Second box: (500-300) * (400-200) = 200 * 200 = 40000
        expected_area_2 = (500 - 300) * (400 - 200)
        assert target["area"][1].item() == expected_area_2
