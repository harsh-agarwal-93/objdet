"""Unit tests for COCO dataset format."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image

from objdet.core.constants import ClassIndexMode
from objdet.core.exceptions import DataFormatError
from objdet.data.formats.coco import COCODataModule, COCODataset


@pytest.fixture
def coco_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary COCO data files."""
    data_dir = tmp_path / "images"
    data_dir.mkdir()

    # Create a dummy image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(data_dir / "img1.jpg")
    img.save(data_dir / "img2.jpg")

    # Create COCO JSON
    ann_file = tmp_path / "instances.json"
    coco_dict = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 10,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 20,
                "bbox": [50, 50, 10, 10],
                "area": 100,
                "iscrowd": 1,
            },
        ],
        "categories": [
            {"id": 10, "name": "cat"},
            {"id": 20, "name": "dog"},
        ],
    }
    with open(ann_file, "w") as f:
        json.dump(coco_dict, f)

    return data_dir, ann_file


def test_coco_dataset_basic(coco_files: tuple[Path, Path]) -> None:
    """Test basic COCODataset functionality."""
    data_dir, ann_file = coco_files
    ds = COCODataset(data_dir, ann_file)

    assert len(ds) == 2
    assert ds.class_names == ["cat", "dog"]
    assert ds.cat_id_to_class_idx[10] == 1  # TorchVision mode
    assert ds.cat_id_to_class_idx[20] == 2

    # Test __getitem__ for image with annotations
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [1, 2]
    assert target["iscrowd"].tolist() == [0, 1]

    # Test __getitem__ for image without annotations
    img2, target2 = ds[1]
    assert target2["boxes"].shape == (0, 4)
    assert target2["labels"].shape == (0,)


def test_coco_dataset_yolo(coco_files: tuple[Path, Path]) -> None:
    """Test COCODataset in YOLO mode."""
    data_dir, ann_file = coco_files
    ds = COCODataset(data_dir, ann_file, class_index_mode=ClassIndexMode.YOLO)

    assert ds.cat_id_to_class_idx[10] == 0
    assert ds.cat_id_to_class_idx[20] == 1


def test_coco_dataset_missing_file(tmp_path: Path) -> None:
    """Test error handling for missing annotation file."""
    with pytest.raises(DataFormatError, match="Annotation file not found"):
        COCODataset(tmp_path, tmp_path / "missing.json")


def test_coco_dataset_invalid_json(tmp_path: Path) -> None:
    """Test error handling for invalid COCO JSON."""
    ann_file = tmp_path / "invalid.json"
    with open(ann_file, "w") as f:
        json.dump({"images": []}, f)  # Missing categories and annotations

    with pytest.raises(DataFormatError, match="Missing required key"):
        COCODataset(tmp_path, ann_file)


def test_coco_datamodule(tmp_path: Path, coco_files: tuple[Path, Path]) -> None:
    """Test COCODataModule integration."""
    data_dir, ann_file = coco_files

    # Create expected directory structure
    (tmp_path / "annotations").mkdir(exist_ok=True)
    shutil.copy(ann_file, tmp_path / "annotations/train.json")
    shutil.copy(ann_file, tmp_path / "annotations/val.json")

    # Create image subdirs (or just use data_dir if we point correctly)
    (tmp_path / "train2017").mkdir(exist_ok=True)
    (tmp_path / "val2017").mkdir(exist_ok=True)
    for f in data_dir.glob("*.jpg"):
        shutil.copy(f, tmp_path / "train2017")
        shutil.copy(f, tmp_path / "val2017")

    dm = COCODataModule(
        data_dir=tmp_path,
        train_ann_file="annotations/train.json",
        val_ann_file="annotations/val.json",
        batch_size=1,
    )

    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.class_names == ["cat", "dog"]

    assert len(dm.train_dataloader()) == 2
    assert len(dm.val_dataloader()) == 2

    # Test test stage
    dm.test_ann_file = tmp_path / "annotations/train.json"  # Reuse train as test
    dm.test_img_dir = tmp_path / "train2017"
    dm.setup("test")
    assert dm.test_dataset is not None

    # Test predict stage
    dm.predict_path = tmp_path / "train2017"
    dm.setup("predict")
    assert dm.predict_dataset is not None
