"""Pascal VOC dataset format parser and DataModule.

This module provides a Dataset and DataModule for Pascal VOC format annotations.
VOC uses XML files per image with the following structure:

<annotation>
    <folder>VOC2012</folder>
    <filename>2007_000027.jpg</filename>
    <size><width>486</width><height>500</height><depth>3</depth></size>
    <object>
        <name>person</name>
        <bndbox><xmin>174</xmin><ymin>101</ymin><xmax>349</xmax><ymax>351</ymax></bndbox>
        <difficult>0</difficult>
    </object>
</annotation>

Example:
    >>> from objdet.data.formats import VOCDataModule
    >>>
    >>> datamodule = VOCDataModule(
    ...     data_dir="/path/to/VOC2012",
    ...     class_names=["aeroplane", "bicycle", "bird", ...],
    ... )
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
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

# Standard Pascal VOC classes
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDataset(Dataset):
    """PyTorch Dataset for Pascal VOC format annotations.

    Args:
        data_dir: Root VOC directory (containing JPEGImages, Annotations).
        split: Dataset split - "train", "val", "trainval", or "test".
        class_names: List of class names. Defaults to VOC classes.
        transforms: Optional transform to apply.
        class_index_mode: How to handle class indices.
        images_dir: Override for images directory (default: data_dir/JPEGImages).
        annotations_dir: Override for annotations directory (default: data_dir/Annotations).
        imagesets_dir: Override for imagesets directory (default: data_dir/ImageSets/Main).

    Attributes:
        image_ids: List of image IDs (filenames without extension).
        class_to_idx: Mapping from class name to index.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "trainval",
        class_names: list[str] | None = None,
        transforms: Callable | None = None,
        class_index_mode: ClassIndexMode = ClassIndexMode.TORCHVISION,
        images_dir: str | Path | None = None,
        annotations_dir: str | Path | None = None,
        imagesets_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.class_names = class_names or VOC_CLASSES
        self.transforms = transforms
        self.class_index_mode = class_index_mode

        # Paths - use overrides if provided, otherwise use standard VOC structure
        self.images_dir = Path(images_dir) if images_dir else self.data_dir / "JPEGImages"
        self.annotations_dir = (
            Path(annotations_dir) if annotations_dir else self.data_dir / "Annotations"
        )
        self.splits_dir = (
            Path(imagesets_dir) if imagesets_dir else self.data_dir / "ImageSets" / "Main"
        )

        # Build class name to index mapping
        self.class_to_idx = {}
        for idx, name in enumerate(self.class_names):
            if self.class_index_mode == ClassIndexMode.TORCHVISION:
                self.class_to_idx[name] = idx + 1  # Background at 0
            else:
                self.class_to_idx[name] = idx

        # Load image IDs from split file
        self._load_split()

        logger.info(
            f"Loaded VOC dataset: {len(self.image_ids)} images, "
            f"{len(self.class_names)} classes, split={split}"
        )

    def _load_split(self) -> None:
        """Load image IDs from split file."""
        split_file = self.splits_dir / f"{self.split}.txt"

        if split_file.exists():
            with open(split_file) as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: use all images in directory
            logger.warning(f"Split file not found: {split_file}. Using all images.")
            self.image_ids = [p.stem for p in self.images_dir.glob("*.jpg")]

        if not self.image_ids:
            raise DataFormatError(
                f"No images found for split '{self.split}'",
                format_name="voc",
                file_path=self.data_dir,
            )

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        """Get image and annotations by index."""
        image_id = self.image_ids[index]

        # Load image
        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.png"

        image = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_path = self.annotations_dir / f"{image_id}.xml"
        target = self._parse_xml(ann_path, image_id)

        # Convert image to tensor
        image_tensor = (
            torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        )

        # Apply transforms
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def _parse_xml(self, xml_path: Path, image_id: str) -> DetectionTarget:
        """Parse VOC XML annotation file."""
        if not xml_path.exists():
            # No annotations for this image
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
                "image_id": hash(image_id) % (2**31),
            }

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficulties = []

        for obj in root.findall("object"):
            # Get class name
            name_elem = obj.find("name")
            if name_elem is None or name_elem.text is None:
                continue
            name = name_elem.text
            if name not in self.class_to_idx:
                logger.warning(f"Unknown class '{name}' in {xml_path}")
                continue

            # Get bounding box
            bbox = self._parse_bbox(obj, xml_path)
            if bbox is None:
                continue

            boxes.append(bbox)
            labels.append(self.class_to_idx[name])

            # Get difficult flag
            difficult = obj.find("difficult")
            difficulties.append(
                int(difficult.text) if difficult is not None and difficult.text is not None else 0
            )

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
                "iscrowd": torch.tensor(difficulties, dtype=torch.int64),
                "image_id": hash(image_id) % (2**31),
            }
        else:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
                "image_id": hash(image_id) % (2**31),
            }

    def _parse_bbox(self, obj: ET.Element, xml_path: Path) -> list[float] | None:
        """Parse bounding box from XML object element."""
        bbox = obj.find("bndbox")
        if bbox is None:
            return None

        parts = {}
        for part in ["xmin", "ymin", "xmax", "ymax"]:
            elem = bbox.find(part)
            if elem is None or elem.text is None:
                return None
            parts[part] = float(elem.text)

        return [parts["xmin"], parts["ymin"], parts["xmax"], parts["ymax"]]


@DATAMODULE_REGISTRY.register("voc", aliases=["pascal_voc"])
class VOCDataModule(BaseDataModule):
    """Lightning DataModule for Pascal VOC format datasets.

    Args:
        data_dir: Root VOC directory.
        train_split: Training split name (default: "trainval").
        val_split: Validation split name (default: "val").
        test_split: Test split name (default: "test").
        **kwargs: Additional arguments for BaseDataModule.
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_split: str = "trainval",
        val_split: str = "val",
        test_split: str = "test",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("dataset_format", DatasetFormat.VOC)
        kwargs.setdefault("class_names", VOC_CLASSES)

        super().__init__(data_dir=data_dir, **kwargs)

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = VOCDataset(
                data_dir=self.data_dir,
                split=self.train_split,
                class_names=self.class_names,
                transforms=self.train_transforms,
                class_index_mode=self.class_index_mode,
            )
            self.val_dataset = VOCDataset(
                data_dir=self.data_dir,
                split=self.val_split,
                class_names=self.class_names,
                transforms=self.val_transforms,
                class_index_mode=self.class_index_mode,
            )

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = VOCDataset(
                data_dir=self.data_dir,
                split=self.val_split,
                class_names=self.class_names,
                transforms=self.val_transforms,
                class_index_mode=self.class_index_mode,
            )

        if stage == "test":
            self.test_dataset = VOCDataset(
                data_dir=self.data_dir,
                split=self.test_split,
                class_names=self.class_names,
                transforms=self.test_transforms,
                class_index_mode=self.class_index_mode,
            )
