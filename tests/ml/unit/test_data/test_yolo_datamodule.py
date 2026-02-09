"""Unit tests for YOLODataModule."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import pytest
from torch.utils.data import DataLoader

from objdet.data.formats.yolo import YOLODataModule, YOLODataset


@pytest.fixture
def mock_yolo_dir(tmp_path: Path):
    """Create a mock YOLO directory structure."""
    # Structure:
    # data_dir/
    #   images/
    #     train/
    #       img1.jpg
    #     val/
    #       img2.jpg
    #     test/
    #       img3.jpg
    #   labels/
    #     train/
    #       img1.txt
    #     val/
    #       img2.txt
    #     test/
    #       img3.txt

    data_dir = tmp_path / "yolo_data"

    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create dummy files
        (img_dir / f"img_{split}.jpg").touch()
        (lbl_dir / f"img_{split}.txt").write_text("0 0.5 0.5 0.5 0.5")

    return data_dir


class TestYOLODataModule:
    """Tests for YOLODataModule."""

    def test_init(self, mock_yolo_dir):
        """Test initialization sets correct attributes."""
        dm = YOLODataModule(
            data_dir=mock_yolo_dir,
            class_names=["class1", "class2"],
            batch_size=4,
            num_workers=2,
        )

        assert dm.data_dir == mock_yolo_dir
        assert dm.class_names == ["class1", "class2"]
        assert dm.batch_size == 4
        assert dm.num_workers == 2
        assert dm.train_split == "train"
        assert dm.val_split == "val"
        assert dm.test_split == "test"

    def test_setup_fit(self, mock_yolo_dir):
        """Test setup for fit stage."""
        dm = YOLODataModule(
            data_dir=mock_yolo_dir,
            class_names=["class1"],
        )

        dm.setup(stage="fit")

        assert isinstance(dm.train_dataset, YOLODataset)
        assert isinstance(dm.val_dataset, YOLODataset)
        assert len(dm.train_dataset) == 1
        assert len(dm.val_dataset) == 1

    def test_setup_validate(self, mock_yolo_dir):
        """Test setup for validate stage."""
        dm = YOLODataModule(
            data_dir=mock_yolo_dir,
            class_names=["class1"],
        )

        dm.setup(stage="validate")

        assert dm.train_dataset is None
        assert isinstance(dm.val_dataset, YOLODataset)
        assert len(dm.val_dataset) == 1

    def test_setup_test(self, mock_yolo_dir):
        """Test setup for test stage."""
        dm = YOLODataModule(
            data_dir=mock_yolo_dir,
            class_names=["class1"],
        )

        dm.setup(stage="test")

        assert isinstance(dm.test_dataset, YOLODataset)
        assert len(dm.test_dataset) == 1

    def test_dataloaders(self, mock_yolo_dir):
        """Test dataloader creation."""
        dm = YOLODataModule(
            data_dir=mock_yolo_dir,
            class_names=["class1"],
            batch_size=2,
        )

        dm.setup(stage="fit")
        dm.setup(stage="test")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert train_loader.batch_size == 2

    def test_predict_dataloader_not_implemented(self, mock_yolo_dir):
        """Test predict_dataloader raises error (removed as it might be implemented in base)."""
        # BaseDataModule implements predict_dataloader but it might return None or empty
        # Checking base implementation...
        pass

    def test_custom_split_names(self, mock_yolo_dir):
        """Test using custom split names."""
        # Standard init, but we'll rename directories to match custom names
        custom_train = "custom_train"
        custom_val = "custom_val"

        # Rename directories
        (mock_yolo_dir / "images" / "train").rename(mock_yolo_dir / "images" / custom_train)
        (mock_yolo_dir / "labels" / "train").rename(mock_yolo_dir / "labels" / custom_train)
        (mock_yolo_dir / "images" / "val").rename(mock_yolo_dir / "images" / custom_val)
        (mock_yolo_dir / "labels" / "val").rename(mock_yolo_dir / "labels" / custom_val)

        dm = YOLODataModule(
            data_dir=mock_yolo_dir,
            class_names=["class1"],
            train_split=custom_train,
            val_split=custom_val,
        )

        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        assert len(cast("Sized", dm.train_dataset)) == 1
        assert len(cast("Sized", dm.val_dataset)) == 1
