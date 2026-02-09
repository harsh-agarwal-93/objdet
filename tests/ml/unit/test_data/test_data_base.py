"""Unit tests for BaseDataModule."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from objdet.core.constants import ClassIndexMode, DatasetFormat
from objdet.core.types import DetectionTarget
from objdet.data.base import BaseDataModule, detection_collate_fn


class MockDataset(Dataset):
    """Simple mock dataset for testing."""

    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        return torch.randn(3, 224, 224), cast(
            "DetectionTarget",
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "labels": torch.tensor([1]),
            },
        )


class MockDataModule(BaseDataModule):
    """Concrete subclass of BaseDataModule for testing."""

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MockDataset(20)
            self.val_dataset = MockDataset(10)
        if stage == "test" or stage is None:
            self.test_dataset = MockDataset(5)
        if stage == "predict" or stage is None:
            self.predict_dataset = MockDataset(2)


@pytest.fixture
def datamodule(tmp_path: Path) -> MockDataModule:
    """Create a MockDataModule instance."""
    return MockDataModule(
        data_dir=tmp_path,
        class_names=["cat", "dog", "bird"],
        batch_size=2,
        num_workers=0,
    )


def test_init(datamodule: MockDataModule, tmp_path: Path) -> None:
    """Test initialization."""
    assert datamodule.data_dir == tmp_path
    assert datamodule.dataset_format == DatasetFormat.COCO
    assert datamodule.class_index_mode == ClassIndexMode.TORCHVISION
    assert datamodule.class_names == ["cat", "dog", "bird"]
    assert datamodule.batch_size == 2
    assert datamodule.num_classes == 3


def test_init_string_enums(tmp_path: Path) -> None:
    """Test init with string enums."""
    dm = MockDataModule(
        data_dir=tmp_path,
        dataset_format="yolo",
        class_index_mode="yolo",
    )
    assert dm.dataset_format == DatasetFormat.YOLO
    assert dm.class_index_mode == ClassIndexMode.YOLO


def test_dataloaders(datamodule: MockDataModule) -> None:
    """Test dataloader creation."""
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    assert len(train_loader) == 10  # 20 / 2

    val_loader = datamodule.val_dataloader()
    assert len(val_loader) == 5  # 10 / 2

    test_loader = datamodule.test_dataloader()
    assert len(test_loader) == 3  # 5 / 2 -> 2.5 ceil is 3? No, DataLoader default.
    # 5 samples, batch 2 -> 3 batches

    predict_loader = datamodule.predict_dataloader()
    assert len(predict_loader) == 1


def test_dataloaders_not_set(datamodule: MockDataModule) -> None:
    """Test error when datasets are not set."""
    with pytest.raises(RuntimeError, match="train_dataset not set"):
        datamodule.train_dataloader()
    with pytest.raises(RuntimeError, match="val_dataset not set"):
        datamodule.val_dataloader()
    with pytest.raises(RuntimeError, match="test_dataset not set"):
        datamodule.test_dataloader()
    with pytest.raises(RuntimeError, match="predict_dataset not set"):
        datamodule.predict_dataloader()


def test_get_class_name(datamodule: MockDataModule) -> None:
    """Test class name retrieval."""
    # TorchVision mode (index 0 is background)
    assert datamodule.get_class_name(0) == "background"
    assert datamodule.get_class_name(1) == "cat"
    assert datamodule.get_class_name(2) == "dog"
    assert datamodule.get_class_name(3) == "bird"
    assert datamodule.get_class_name(4) == "unknown_3"  # Index 4 -> class_id 3 -> out of bounds

    # YOLO mode
    datamodule.class_index_mode = ClassIndexMode.YOLO
    assert datamodule.get_class_name(0) == "cat"
    assert datamodule.get_class_name(1) == "dog"
    assert datamodule.get_class_name(2) == "bird"
    assert datamodule.get_class_name(3) == "unknown_3"


def test_get_dataset_info(datamodule: MockDataModule) -> None:
    """Test dataset info dictionary."""
    datamodule.setup()
    info = datamodule.get_dataset_info()

    assert info["num_classes"] == 3
    assert info["train_samples"] == 20
    assert info["val_samples"] == 10
    assert info["test_samples"] == 5
    assert info["batch_size"] == 2


def test_detection_collate_fn() -> None:
    """Test the detection collate function."""
    batch = [
        (
            torch.randn(3, 10, 10),
            cast(
                "DetectionTarget",
                {"boxes": torch.tensor([[0, 0, 1, 1]]), "labels": torch.tensor([1])},
            ),
        ),
        (
            torch.randn(3, 10, 10),
            cast(
                "DetectionTarget",
                {"boxes": torch.tensor([[2, 2, 3, 3]]), "labels": torch.tensor([2])},
            ),
        ),
    ]
    images, targets = detection_collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2
    assert isinstance(images[0], Tensor)
    assert targets[1]["labels"][0] == 2
