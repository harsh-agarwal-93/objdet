"""Base DataModule for object detection datasets.

This module provides the abstract base class for all detection data modules.
It defines the common interface for data loading, augmentation, and batching.

Example:
    >>> from objdet.data.base import BaseDataModule
    >>>
    >>> class COCODataModule(BaseDataModule):
    ...     def setup(self, stage: str | None = None):
    ...         if stage == "fit":
    ...             self.train_dataset = COCODataset(self.data_dir / "train")
    ...             self.val_dataset = COCODataset(self.data_dir / "val")
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from objdet.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    ClassIndexMode,
    DatasetFormat,
)
from objdet.core.logging import get_logger
from objdet.core.types import DetectionTarget

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def detection_collate_fn(
    batch: list[tuple[Tensor, DetectionTarget]],
) -> tuple[list[Tensor], list[DetectionTarget]]:
    """Collate function for detection batches.

    Unlike classification where we can stack images, detection requires
    lists because targets have variable numbers of boxes.

    Args:
        batch: List of (image, target) tuples.

    Returns:
        Tuple of (list of images, list of targets).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class BaseDataModule(L.LightningDataModule):
    """Abstract base class for detection data modules.

    This class provides common functionality for all detection datasets:
    - Standard DataLoader configuration
    - Train/val/test/predict dataset management
    - Augmentation pipeline integration
    - Class index mode handling

    Subclasses must implement:
    - `setup()`: Create datasets for each stage

    Args:
        data_dir: Root directory containing the dataset.
        dataset_format: Format of the dataset annotations.
        class_index_mode: How class indices should be handled.
        class_names: List of class names (excluding background for YOLO,
            including all classes for TorchVision).
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoaders.
        pin_memory: Whether to pin memory in DataLoaders.
        train_transforms: Transform pipeline for training data.
        val_transforms: Transform pipeline for validation data.
        test_transforms: Transform pipeline for test data.

    Attributes:
        data_dir: Dataset root directory.
        dataset_format: Dataset format enum.
        class_index_mode: Class index handling mode.
        class_names: List of class names.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.

    Example:
        >>> datamodule = COCODataModule(
        ...     data_dir="/path/to/coco",
        ...     class_index_mode=ClassIndexMode.TORCHVISION,
        ...     batch_size=16,
        ... )
        >>> datamodule.setup("fit")
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_format: DatasetFormat | str = DatasetFormat.COCO,
        class_index_mode: ClassIndexMode | str = ClassIndexMode.TORCHVISION,
        class_names: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        predict_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        # Convert strings to enums
        if isinstance(dataset_format, str):
            dataset_format = DatasetFormat(dataset_format)
        if isinstance(class_index_mode, str):
            class_index_mode = ClassIndexMode(class_index_mode)

        self.data_dir = Path(data_dir)
        self.dataset_format = dataset_format
        self.class_index_mode = class_index_mode
        self.class_names = class_names or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.predict_path = Path(predict_path) if predict_path else None

        # Datasets (set in setup())  # noqa: ERA001
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None

        # Save hyperparameters for reproducibility
        self.save_hyperparameters(ignore=["train_transforms", "val_transforms", "test_transforms"])

        logger.info(
            f"Initialized {self.__class__.__name__}",
            data_dir=str(self.data_dir),
            dataset_format=dataset_format.value,
            class_index_mode=class_index_mode.value,
            batch_size=batch_size,
        )

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training, validation, testing, or prediction.

        This method must be implemented by subclasses to create the
        appropriate datasets for each stage.

        Args:
            stage: One of "fit", "validate", "test", or "predict".
                   If None, set up all datasets.
        """
        ...

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            DataLoader for training data with shuffling enabled.
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not set. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=detection_collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            DataLoader for validation data without shuffling.
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not set. Call setup('fit' or 'validate') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=detection_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Returns:
            DataLoader for test data without shuffling.
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not set. Call setup('test') first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader.

        Returns:
            DataLoader for prediction data without shuffling.
        """
        if self.predict_dataset is None:
            raise RuntimeError("predict_dataset not set. Call setup('predict') first.")

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate_fn,
        )

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset.

        Returns:
            Number of classes (not including background for YOLO mode).
        """
        return len(self.class_names)

    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID.

        Args:
            class_id: Class index (accounting for class_index_mode).

        Returns:
            Class name string.
        """
        # Adjust for background class if TorchVision mode
        if self.class_index_mode == ClassIndexMode.TORCHVISION:
            if class_id == 0:
                return "background"
            class_id -= 1

        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"unknown_{class_id}"

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information dictionary.

        Returns:
            Dictionary with dataset metadata.
        """
        info = {
            "data_dir": str(self.data_dir),
            "dataset_format": self.dataset_format.value,
            "class_index_mode": self.class_index_mode.value,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "batch_size": self.batch_size,
        }

        if self.train_dataset is not None:
            info["train_samples"] = len(self.train_dataset)  # type: ignore[arg-type]
        if self.val_dataset is not None:
            info["val_samples"] = len(self.val_dataset)  # type: ignore[arg-type]
        if self.test_dataset is not None:
            info["test_samples"] = len(self.test_dataset)  # type: ignore[arg-type]

        return info
