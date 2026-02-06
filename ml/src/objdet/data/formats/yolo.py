"""YOLO dataset format parser and DataModule.

This module provides a Dataset and DataModule for YOLO format annotations.
YOLO uses one text file per image with the following structure:

<class_id> <x_center> <y_center> <width> <height>

Where coordinates are normalized to [0, 1] relative to image dimensions.

Example:
    >>> from objdet.data.formats import YOLODataModule
    >>>
    >>> datamodule = YOLODataModule(
    ...     data_dir="/path/to/yolo_dataset",
    ...     class_names=["person", "car", "dog"],
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from objdet.core.constants import ClassIndexMode, DatasetFormat
from objdet.core.exceptions import DataFormatError
from objdet.core.logging import get_logger
from objdet.data.base import BaseDataModule
from objdet.data.registry import DATAMODULE_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from objdet.core.types import DetectionTarget

logger = get_logger(__name__)


class YOLODataset(Dataset):
    """PyTorch Dataset for YOLO format annotations.

    Directory structure expected:
    data_dir/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── val/
    └── labels/
        ├── train/
        │   ├── img1.txt
        │   └── ...
        └── val/

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing label text files.
        class_names: List of class names (required for YOLO).
        transforms: Optional transform to apply.
        class_index_mode: How to handle class indices.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: list[str],
        transforms: Callable | None = None,
        class_index_mode: ClassIndexMode = ClassIndexMode.YOLO,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.transforms = transforms
        self.class_index_mode = class_index_mode

        # Find all images
        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_paths.extend(self.images_dir.glob(ext))
            self.image_paths.extend(self.images_dir.glob(ext.upper()))

        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            raise DataFormatError(
                f"No images found in {self.images_dir}",
                format_name="yolo",
                file_path=self.images_dir,
            )

        logger.info(
            f"Loaded YOLO dataset: {len(self.image_paths)} images, {len(self.class_names)} classes"
        )

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        """Get image and annotations by index."""
        img_path = self.image_paths[index]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # Find corresponding label file
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        target = self._parse_label_file(label_path, img_width, img_height, index)

        # Convert image to tensor
        image_tensor = (
            torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        )

        # Apply transforms
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def _parse_label_file(
        self,
        label_path: Path,
        img_width: int,
        img_height: int,
        idx: int,
    ) -> DetectionTarget:
        """Parse YOLO format label file."""
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    # Parse YOLO format: class_id x_center y_center width height
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert normalized coords to pixel coords (xyxy format)
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height

                    # Clamp to image bounds
                    x1 = max(0.0, min(x1, img_width))
                    y1 = max(0.0, min(y1, img_height))
                    x2 = max(0.0, min(x2, img_width))
                    y2 = max(0.0, min(y2, img_height))

                    boxes.append([x1, y1, x2, y2])

                    # Handle class index mode
                    if self.class_index_mode == ClassIndexMode.TORCHVISION:
                        labels.append(class_id + 1)  # Add 1 for background
                    else:
                        labels.append(class_id)

        # Convert to tensors
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )

            return {
                "boxes": boxes_tensor,
                "labels": torch.tensor(labels, dtype=torch.int64),
                "area": areas,
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                "image_id": idx,
            }
        else:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
                "image_id": idx,
            }


@DATAMODULE_REGISTRY.register("yolo", aliases=["yolo_txt", "ultralytics"])
class YOLODataModule(BaseDataModule):
    """Lightning DataModule for YOLO format datasets.

    Expected directory structure::

        data_dir/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

    Args:
        data_dir: Root dataset directory.
        images_subdir: Subdirectory name for images (default: "images").
        labels_subdir: Subdirectory name for labels (default: "labels").
        train_split: Training split name (default: "train").
        val_split: Validation split name (default: "val").
        **kwargs: Additional arguments for BaseDataModule.
    """

    def __init__(
        self,
        data_dir: str | Path,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("dataset_format", DatasetFormat.YOLO)
        kwargs.setdefault("class_index_mode", ClassIndexMode.YOLO)

        super().__init__(data_dir=data_dir, **kwargs)

        self.images_subdir = images_subdir
        self.labels_subdir = labels_subdir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = YOLODataset(
                images_dir=self.data_dir / self.images_subdir / self.train_split,
                labels_dir=self.data_dir / self.labels_subdir / self.train_split,
                class_names=self.class_names,
                transforms=self.train_transforms,
                class_index_mode=self.class_index_mode,
            )
            self.val_dataset = YOLODataset(
                images_dir=self.data_dir / self.images_subdir / self.val_split,
                labels_dir=self.data_dir / self.labels_subdir / self.val_split,
                class_names=self.class_names,
                transforms=self.val_transforms,
                class_index_mode=self.class_index_mode,
            )

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = YOLODataset(
                images_dir=self.data_dir / self.images_subdir / self.val_split,
                labels_dir=self.data_dir / self.labels_subdir / self.val_split,
                class_names=self.class_names,
                transforms=self.val_transforms,
                class_index_mode=self.class_index_mode,
            )

        if stage == "test":
            test_dir = self.data_dir / self.images_subdir / self.test_split
            if test_dir.exists():
                self.test_dataset = YOLODataset(
                    images_dir=test_dir,
                    labels_dir=self.data_dir / self.labels_subdir / self.test_split,
                    class_names=self.class_names,
                    transforms=self.test_transforms,
                    class_index_mode=self.class_index_mode,
                )
