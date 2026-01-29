"""COCO dataset format parser and DataModule.

This module provides a Dataset and DataModule for COCO-format annotations.
COCO uses a single JSON file containing all annotations with the following structure:

{
    "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}, ...],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], ...}, ...],
    "categories": [{"id": 1, "name": "cat"}, ...]
}

Example:
    >>> from objdet.data.formats import COCODataModule
    >>>
    >>> datamodule = COCODataModule(
    ...     data_dir="/path/to/coco",
    ...     train_ann_file="annotations/instances_train2017.json",
    ...     val_ann_file="annotations/instances_val2017.json",
    ... )
"""

from __future__ import annotations

import json
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


class COCODataset(Dataset):
    """PyTorch Dataset for COCO format annotations.

    Args:
        data_dir: Root directory containing images.
        ann_file: Path to annotation JSON file.
        transforms: Optional transform to apply to images and targets.
        class_index_mode: How to handle class indices.

    Attributes:
        images: List of image metadata dicts.
        annotations: Dict mapping image_id to list of annotations.
        categories: Dict mapping category_id to category info.
        class_names: List of class names in order.
    """

    def __init__(
        self,
        data_dir: str | Path,
        ann_file: str | Path,
        transforms: Callable | None = None,
        class_index_mode: ClassIndexMode = ClassIndexMode.TORCHVISION,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        self.class_index_mode = class_index_mode

        # Load annotations
        self._load_annotations()

        logger.info(
            f"Loaded COCO dataset: {len(self.images)} images, {len(self.class_names)} classes"
        )

    def _load_annotations(self) -> None:
        """Load and parse COCO annotation file."""
        if not self.ann_file.exists():
            raise DataFormatError(
                f"Annotation file not found: {self.ann_file}",
                format_name="coco",
                file_path=self.ann_file,
            )

        with open(self.ann_file) as f:
            coco_data = json.load(f)

        # Validate required keys
        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in coco_data:
                raise DataFormatError(
                    f"Missing required key '{key}' in COCO annotation file",
                    format_name="coco",
                    file_path=self.ann_file,
                )

        # Store images
        self.images = coco_data["images"]

        # Build image_id to filename mapping
        self.image_id_to_info = {img["id"]: img for img in self.images}

        # Build category mapping
        # Sort categories by id to ensure consistent ordering
        sorted_categories = sorted(coco_data["categories"], key=lambda x: x["id"])
        self.categories = {cat["id"]: cat for cat in sorted_categories}

        # Create contiguous class index mapping
        # COCO category IDs are not necessarily contiguous (1-90 with gaps)
        self.cat_id_to_class_idx = {}
        self.class_names = []

        for idx, cat in enumerate(sorted_categories):
            # For TorchVision: class indices start at 1 (0 is background)
            # For YOLO: class indices start at 0
            if self.class_index_mode == ClassIndexMode.TORCHVISION:
                class_idx = idx + 1
            else:
                class_idx = idx

            self.cat_id_to_class_idx[cat["id"]] = class_idx
            self.class_names.append(cat["name"])

        # Group annotations by image_id
        self.annotations: dict[int, list[dict]] = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        """Get image and annotations by index.

        Args:
            index: Dataset index.

        Returns:
            Tuple of (image_tensor, target_dict).
        """
        img_info = self.images[index]
        image_id = img_info["id"]

        # Load image
        img_path = self.data_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.annotations.get(image_id, [])

        # Build target dict
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # Skip crowd annotations if needed
            if ann.get("iscrowd", 0):
                iscrowd.append(1)
            else:
                iscrowd.append(0)

            # Convert bbox from [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            # Map category_id to class index
            class_idx = self.cat_id_to_class_idx[ann["category_id"]]
            labels.append(class_idx)

            areas.append(ann.get("area", w * h))

        # Convert to tensors
        if boxes:
            target: DetectionTarget = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
                "image_id": image_id,
            }
        else:
            # No annotations for this image
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
                "image_id": image_id,
            }

        # Convert image to tensor
        image_tensor = (
            torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        )

        # Apply transforms if provided
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


@DATAMODULE_REGISTRY.register("coco")
class COCODataModule(BaseDataModule):
    """Lightning DataModule for COCO format datasets.

    Args:
        data_dir: Root directory containing images.
        train_ann_file: Path to training annotations JSON.
        val_ann_file: Path to validation annotations JSON.
        test_ann_file: Optional path to test annotations JSON.
        train_img_dir: Subdirectory for training images (relative to data_dir).
        val_img_dir: Subdirectory for validation images.
        test_img_dir: Subdirectory for test images.
        **kwargs: Additional arguments for BaseDataModule.

    Example:
        >>> datamodule = COCODataModule(
        ...     data_dir="/data/coco",
        ...     train_ann_file="annotations/instances_train2017.json",
        ...     val_ann_file="annotations/instances_val2017.json",
        ...     train_img_dir="train2017",
        ...     val_img_dir="val2017",
        ...     batch_size=16,
        ... )
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_ann_file: str = "annotations/instances_train2017.json",
        val_ann_file: str = "annotations/instances_val2017.json",
        test_ann_file: str | None = None,
        train_img_dir: str = "train2017",
        val_img_dir: str = "val2017",
        test_img_dir: str = "test2017",
        **kwargs: Any,
    ) -> None:
        # Set default format
        kwargs.setdefault("dataset_format", DatasetFormat.COCO)

        super().__init__(data_dir=data_dir, **kwargs)

        self.train_ann_file = self.data_dir / train_ann_file
        self.val_ann_file = self.data_dir / val_ann_file
        self.test_ann_file = self.data_dir / test_ann_file if test_ann_file else None
        self.train_img_dir = self.data_dir / train_img_dir
        self.val_img_dir = self.data_dir / val_img_dir
        self.test_img_dir = self.data_dir / test_img_dir

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: One of "fit", "validate", "test", or "predict".
        """
        if stage == "fit" or stage is None:
            self.train_dataset = COCODataset(
                data_dir=self.train_img_dir,
                ann_file=self.train_ann_file,
                transforms=self.train_transforms,
                class_index_mode=self.class_index_mode,
            )
            self.val_dataset = COCODataset(
                data_dir=self.val_img_dir,
                ann_file=self.val_ann_file,
                transforms=self.val_transforms,
                class_index_mode=self.class_index_mode,
            )

            # Set class names from dataset
            if not self.class_names:
                self.class_names = self.train_dataset.class_names

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = COCODataset(
                data_dir=self.val_img_dir,
                ann_file=self.val_ann_file,
                transforms=self.val_transforms,
                class_index_mode=self.class_index_mode,
            )

        if stage == "test" and self.test_ann_file is not None:
            self.test_dataset = COCODataset(
                data_dir=self.test_img_dir,
                ann_file=self.test_ann_file,
                transforms=self.test_transforms,
                class_index_mode=self.class_index_mode,
            )

        if stage == "predict" and self.predict_path is not None:
            # For prediction, create a simple image-only dataset
            from objdet.data.formats.predict import PredictDataset

            self.predict_dataset = PredictDataset(
                image_dir=self.predict_path,
                transforms=self.val_transforms,
            )
