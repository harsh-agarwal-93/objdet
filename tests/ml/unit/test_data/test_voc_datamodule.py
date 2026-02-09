from pathlib import Path
from typing import cast

import pytest
from torch.utils.data import DataLoader

from objdet.data.formats.voc import VOCDataModule, VOCDataset


@pytest.fixture
def mock_voc_dir(tmp_path: Path):
    """Create a mock VOC dataset directory structure."""
    data_dir = tmp_path / "VOC2012"
    img_dir = data_dir / "JPEGImages"
    ann_dir = data_dir / "Annotations"
    splits_dir = data_dir / "ImageSets" / "Main"

    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    # Create dummy files
    for name in ["img1", "img2", "img3"]:
        (img_dir / f"{name}.jpg").touch()
        (ann_dir / f"{name}.xml").write_text("<annotation></annotation>")

    (splits_dir / "trainval.txt").write_text("img1\n")
    (splits_dir / "val.txt").write_text("img2\n")
    (splits_dir / "test.txt").write_text("img3\n")

    return data_dir


class TestVOCDataModule:
    """Tests for VOCDataModule."""

    def test_init(self, mock_voc_dir):
        """Test initialization sets correct attributes."""
        dm = VOCDataModule(
            data_dir=mock_voc_dir,
            batch_size=4,
            num_workers=2,
        )

        assert dm.data_dir == mock_voc_dir
        assert dm.batch_size == 4
        assert dm.num_workers == 2
        assert dm.train_split == "trainval"
        assert dm.val_split == "val"
        assert dm.test_split == "test"

        # Test defaults
        assert "person" in dm.class_names

    def test_setup_fit(self, mock_voc_dir):
        """Test setup for fit stage."""
        dm = VOCDataModule(
            data_dir=mock_voc_dir,
        )

        dm.setup(stage="fit")

        assert isinstance(dm.train_dataset, VOCDataset)
        assert isinstance(dm.val_dataset, VOCDataset)
        assert len(dm.train_dataset) == 1
        assert len(dm.val_dataset) == 1
        # Check specific IDs loaded
        assert dm.train_dataset.image_ids == ["img1"]
        assert dm.val_dataset.image_ids == ["img2"]

    def test_setup_validate(self, mock_voc_dir):
        """Test setup for validate stage."""
        dm = VOCDataModule(
            data_dir=mock_voc_dir,
        )

        dm.setup(stage="validate")

        assert dm.train_dataset is None
        assert isinstance(dm.val_dataset, VOCDataset)
        assert len(dm.val_dataset) == 1

    def test_setup_test(self, mock_voc_dir):
        """Test setup for test stage."""
        dm = VOCDataModule(
            data_dir=mock_voc_dir,
        )

        dm.setup(stage="test")

        assert isinstance(dm.test_dataset, VOCDataset)
        assert len(dm.test_dataset) == 1
        assert dm.test_dataset.image_ids == ["img3"]

    def test_dataloaders(self, mock_voc_dir):
        """Test dataloader creation."""
        dm = VOCDataModule(
            data_dir=mock_voc_dir,
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

    def test_custom_class_names(self, mock_voc_dir):
        """Test using custom class names."""
        custom_classes = ["foo", "bar"]
        dm = VOCDataModule(
            data_dir=mock_voc_dir,
            class_names=custom_classes,
        )

        assert dm.class_names == custom_classes

        # Setup and check dataset uses custom classes
        dm.setup("fit")
        assert dm.train_dataset is not None
        assert cast("VOCDataset", dm.train_dataset).class_names == custom_classes
