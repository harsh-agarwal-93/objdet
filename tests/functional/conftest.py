"""Fixtures for functional tests.

This module provides fixtures for end-to-end testing of ObjDet workflows,
including synthetic dataset generation and temporary configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_coco_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a minimal COCO-format dataset with synthetic images.

    This fixture creates:
    - 4 synthetic RGB images (128x128) with colored rectangles
    - COCO JSON annotations with bounding boxes
    - train/val splits

    Returns:
        Path to the dataset root directory.
    """
    dataset_dir = tmp_path_factory.mktemp("sample_coco")

    # Create directory structure
    train_images_dir = dataset_dir / "train2017"
    val_images_dir = dataset_dir / "val2017"
    annotations_dir = dataset_dir / "annotations"

    train_images_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    # Define class names
    categories = [
        {"id": 1, "name": "rectangle", "supercategory": "shape"},
        {"id": 2, "name": "square", "supercategory": "shape"},
    ]

    def create_synthetic_image(
        image_id: int,
        width: int = 128,
        height: int = 128,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Create a synthetic image with random colored rectangles."""
        # Create base image with random background
        rng = np.random.default_rng(image_id)
        img = rng.integers(50, 150, (height, width, 3), dtype=np.uint8)

        annotations = []
        annotation_id_offset = image_id * 10

        # Add 1-3 rectangles per image
        num_objects = rng.integers(1, 4)
        for obj_idx in range(num_objects):
            # Random rectangle position and size
            x = int(rng.integers(10, width - 50))
            y = int(rng.integers(10, height - 50))
            w = int(rng.integers(20, min(50, width - x)))
            h = int(rng.integers(20, min(50, height - y)))

            # Random bright color for the rectangle
            color = rng.integers(180, 255, 3)

            # Draw rectangle
            img[y : y + h, x : x + w] = color

            # Create COCO annotation
            category_id = 1 if w != h else 2  # rectangle vs square
            annotations.append(
                {
                    "id": annotation_id_offset + obj_idx,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],  # COCO format: [x, y, width, height]
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

        return img, annotations

    # Generate training images (3 images)
    train_images = []
    train_annotations = []
    for i in range(1, 4):
        img, anns = create_synthetic_image(i)
        filename = f"image_{i:04d}.jpg"
        Image.fromarray(img).save(train_images_dir / filename)
        train_images.append(
            {
                "id": i,
                "file_name": filename,
                "width": 128,
                "height": 128,
            }
        )
        train_annotations.extend(anns)

    # Generate validation images (2 images)
    val_images = []
    val_annotations = []
    for i in range(4, 6):
        img, anns = create_synthetic_image(i)
        filename = f"image_{i:04d}.jpg"
        Image.fromarray(img).save(val_images_dir / filename)
        val_images.append(
            {
                "id": i,
                "file_name": filename,
                "width": 128,
                "height": 128,
            }
        )
        val_annotations.extend(anns)

    # Create COCO annotation files
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }

    with open(annotations_dir / "instances_train2017.json", "w") as f:
        json.dump(train_coco, f, indent=2)

    with open(annotations_dir / "instances_val2017.json", "w") as f:
        json.dump(val_coco, f, indent=2)

    return dataset_dir


@pytest.fixture(scope="session")
def sample_litdata_dataset(
    sample_coco_dataset: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Convert sample COCO dataset to LitData format.

    Returns:
        Path to the LitData dataset directory.
    """
    litdata_dir = tmp_path_factory.mktemp("sample_litdata")

    # Import preprocessing module
    from objdet.data.preprocessing import convert_to_litdata

    # Convert train split
    convert_to_litdata(
        output_dir=str(litdata_dir),
        format_name="coco",
        coco_ann_file=str(sample_coco_dataset / "annotations" / "instances_train2017.json"),
        coco_images_dir=str(sample_coco_dataset / "train2017"),
        splits=["train"],
        num_workers=1,
    )

    # Convert val split
    convert_to_litdata(
        output_dir=str(litdata_dir),
        format_name="coco",
        coco_ann_file=str(sample_coco_dataset / "annotations" / "instances_val2017.json"),
        coco_images_dir=str(sample_coco_dataset / "val2017"),
        splits=["val"],
        num_workers=1,
    )

    return litdata_dir


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def faster_rcnn_config(sample_coco_dataset: Path, tmp_path: Path) -> Path:
    """Create a Faster R-CNN config for testing with sample COCO data.

    Returns:
        Path to the configuration YAML file.
    """
    config = {
        "model": {
            "class_path": "objdet.models.torchvision.FasterRCNN",
            "init_args": {
                "num_classes": 2,
                "backbone": "resnet50_fpn",
                "pretrained": False,
                "pretrained_backbone": False,  # Faster for tests
                "min_size": 128,
                "max_size": 256,
                "learning_rate": 0.001,
            },
        },
        "data": {
            "class_path": "objdet.data.formats.coco.COCODataModule",
            "init_args": {
                "data_dir": str(sample_coco_dataset),
                "train_ann_file": "annotations/instances_train2017.json",
                "val_ann_file": "annotations/instances_val2017.json",
                "train_img_dir": "train2017",
                "val_img_dir": "val2017",
                "batch_size": 2,
                "num_workers": 0,
            },
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
    }

    config_path = tmp_path / "faster_rcnn_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def retinanet_config(sample_coco_dataset: Path, tmp_path: Path) -> Path:
    """Create a RetinaNet config for testing with sample COCO data.

    Returns:
        Path to the configuration YAML file.
    """
    config = {
        "model": {
            "class_path": "objdet.models.torchvision.RetinaNet",
            "init_args": {
                "num_classes": 2,
                "backbone": "resnet50_fpn",
                "pretrained": False,
                "pretrained_backbone": False,
                "min_size": 128,
                "max_size": 256,
                "learning_rate": 0.001,
            },
        },
        "data": {
            "class_path": "objdet.data.formats.coco.COCODataModule",
            "init_args": {
                "data_dir": str(sample_coco_dataset),
                "train_ann_file": "annotations/instances_train2017.json",
                "val_ann_file": "annotations/instances_val2017.json",
                "train_img_dir": "train2017",
                "val_img_dir": "val2017",
                "batch_size": 2,
                "num_workers": 0,
            },
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
    }

    config_path = tmp_path / "retinanet_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def yolov8_config(sample_coco_dataset: Path, tmp_path: Path) -> Path:
    """Create a YOLOv8 config for testing with sample COCO data.

    Returns:
        Path to the configuration YAML file.
    """
    config = {
        "model": {
            "class_path": "objdet.models.yolo.YOLOv8",
            "init_args": {
                "num_classes": 2,
                "model_size": "n",  # Nano for fast testing
                "pretrained": False,
                "learning_rate": 0.001,
            },
        },
        "data": {
            "class_path": "objdet.data.formats.coco.COCODataModule",
            "init_args": {
                "data_dir": str(sample_coco_dataset),
                "train_ann_file": "annotations/instances_train2017.json",
                "val_ann_file": "annotations/instances_val2017.json",
                "train_img_dir": "train2017",
                "val_img_dir": "val2017",
                "batch_size": 2,
                "num_workers": 0,
                "class_index_mode": "yolo",
            },
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
    }

    config_path = tmp_path / "yolov8_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def litdata_config(sample_litdata_dataset: Path, tmp_path: Path) -> Path:
    """Create a config for testing with LitData dataset.

    Returns:
        Path to the configuration YAML file.
    """
    config = {
        "model": {
            "class_path": "objdet.models.torchvision.FasterRCNN",
            "init_args": {
                "num_classes": 2,
                "backbone": "resnet50_fpn",
                "pretrained": False,
                "pretrained_backbone": False,
                "min_size": 128,
                "max_size": 256,
                "learning_rate": 0.001,
            },
        },
        "data": {
            "class_path": "objdet.data.formats.litdata.LitDataDataModule",
            "init_args": {
                "data_dir": str(sample_litdata_dataset),
                "train_subdir": "train",
                "val_subdir": "val",
                "batch_size": 2,
                "num_workers": 0,
            },
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
    }

    config_path = tmp_path / "litdata_test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for training artifacts.

    Returns:
        Path to the temporary output directory.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def cleanup_cuda_functional() -> Generator[None, None, None]:
    """Clean up CUDA memory after each functional test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
