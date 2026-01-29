"""LitData optimized dataset format for fast streaming.

This module provides Dataset and DataModule classes for consuming
datasets that have been preprocessed into LitData's optimized streaming format.

LitData enables:
- Fast random access to preprocessed data
- Efficient streaming from cloud storage
- Automatic batching and prefetching

Example:
    >>> from objdet.data.formats.litdata import LitDataDataModule
    >>>
    >>> # Use preprocessed LitData format
    >>> datamodule = LitDataDataModule(
    ...     data_dir="/data/coco_litdata",  # Contains train/, val/ subdirs
    ...     batch_size=16,
    ... )
    >>> datamodule.setup("fit")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from objdet.core.constants import ClassIndexMode, DatasetFormat
from objdet.core.exceptions import DataError, DependencyError
from objdet.core.logging import get_logger
from objdet.data.base import BaseDataModule
from objdet.data.registry import DATAMODULE_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from objdet.core.types import DetectionTarget

logger = get_logger(__name__)


class LitDataDataset(Dataset):
    """Dataset for reading LitData optimized format.

    This class wraps LitData's StreamingDataset to provide a detection-compatible
    interface that returns (image_tensor, target_dict) tuples.

    Args:
        data_dir: Directory containing LitData optimized chunks.
        transforms: Optional transform to apply to (image, target) pairs.
        shuffle: Whether to shuffle the data.
        drop_last: Whether to drop the last incomplete batch.

    Attributes:
        data_dir: Path to the data directory.
        transforms: Transform function if provided.

    Example:
        >>> dataset = LitDataDataset(
        ...     data_dir="/data/coco_litdata/train",
        ...     shuffle=True,
        ... )
        >>> image, target = dataset[0]
    """

    def __init__(
        self,
        data_dir: str | Path,
        transforms: Callable | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        try:
            from litdata import StreamingDataset
        except ImportError as e:
            raise DependencyError(
                "LitData is required for streaming datasets",
                package_name="litdata",
                install_command="uv add litdata",
            ) from e

        self.data_dir = Path(data_dir)
        self.transforms = transforms

        if not self.data_dir.exists():
            raise DataError(f"Data directory not found: {self.data_dir}")

        # Initialize the underlying StreamingDataset
        self._dataset = StreamingDataset(
            input_dir=str(self.data_dir),
            shuffle=shuffle,
            drop_last=drop_last,
        )

        logger.info(
            f"Loaded LitData dataset from {self.data_dir}",
            num_samples=len(self._dataset),
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        """Get a sample by index.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_tensor, target_dict).
        """
        sample = self._dataset[index]

        # Convert numpy arrays to tensors
        image = torch.from_numpy(sample["image"]).float()
        target: DetectionTarget = {
            "boxes": torch.from_numpy(sample["boxes"]).float(),
            "labels": torch.from_numpy(sample["labels"]).long(),
            "area": torch.from_numpy(sample["area"]).float(),
            "image_id": sample["image_id"],
        }

        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


@DATAMODULE_REGISTRY.register("litdata", aliases=["streaming"])
class LitDataDataModule(BaseDataModule):
    """Lightning DataModule for LitData optimized datasets.

    This DataModule uses LitData's StreamingDataLoader for efficient
    data loading from preprocessed datasets.

    Args:
        data_dir: Root directory containing train/val/test subdirectories.
        train_subdir: Subdirectory name for training data.
        val_subdir: Subdirectory name for validation data.
        test_subdir: Subdirectory name for test data.
        **kwargs: Additional arguments for BaseDataModule.

    Example:
        >>> datamodule = LitDataDataModule(
        ...     data_dir="/data/coco_litdata",
        ...     batch_size=16,
        ...     num_workers=4,
        ... )
        >>> datamodule.setup("fit")
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_subdir: str = "train",
        val_subdir: str = "val",
        test_subdir: str = "test",
        class_names: list[str] | None = None,
        class_index_mode: ClassIndexMode | str = ClassIndexMode.TORCHVISION,
        **kwargs: Any,
    ) -> None:
        # Set dataset format to LITDATA (we'll add this enum value)
        # For now, use COCO as a fallback since LitData is format-agnostic
        kwargs.setdefault("dataset_format", DatasetFormat.COCO)

        super().__init__(
            data_dir=data_dir,
            class_names=class_names,
            class_index_mode=class_index_mode,
            **kwargs,
        )

        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
        self.test_subdir = test_subdir

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: One of "fit", "validate", "test", or "predict".
        """
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / self.train_subdir
            if train_dir.exists():
                self.train_dataset = LitDataDataset(
                    data_dir=train_dir,
                    transforms=self.train_transforms,
                    shuffle=True,
                    drop_last=True,
                )

            val_dir = self.data_dir / self.val_subdir
            if val_dir.exists():
                self.val_dataset = LitDataDataset(
                    data_dir=val_dir,
                    transforms=self.val_transforms,
                    shuffle=False,
                )

        if stage == "validate" and self.val_dataset is None:
            val_dir = self.data_dir / self.val_subdir
            if val_dir.exists():
                self.val_dataset = LitDataDataset(
                    data_dir=val_dir,
                    transforms=self.val_transforms,
                    shuffle=False,
                )

        if stage == "test":
            test_dir = self.data_dir / self.test_subdir
            if test_dir.exists():
                self.test_dataset = LitDataDataset(
                    data_dir=test_dir,
                    transforms=self.test_transforms,
                    shuffle=False,
                )

    def train_dataloader(self) -> Any:
        """Create training DataLoader using LitData's StreamingDataLoader.

        Returns:
            StreamingDataLoader for training data with shuffling enabled.
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not set. Call setup('fit') first.")

        try:
            from litdata import StreamingDataLoader
        except ImportError as e:
            raise DependencyError(
                "LitData is required for streaming datasets",
                package_name="litdata",
                install_command="uv add litdata",
            ) from e

        return StreamingDataLoader(
            self.train_dataset._dataset,  # type: ignore[attr-defined] # Use underlying StreamingDataset
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> Any:
        """Create validation DataLoader using LitData's StreamingDataLoader.

        Returns:
            StreamingDataLoader for validation data without shuffling.
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not set. Call setup('fit' or 'validate') first.")

        try:
            from litdata import StreamingDataLoader
        except ImportError as e:
            raise DependencyError(
                "LitData is required for streaming datasets",
                package_name="litdata",
                install_command="uv add litdata",
            ) from e

        return StreamingDataLoader(
            self.val_dataset._dataset,  # type: ignore[attr-defined]
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Any:
        """Create test DataLoader using LitData's StreamingDataLoader.

        Returns:
            StreamingDataLoader for test data without shuffling.
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not set. Call setup('test') first.")

        try:
            from litdata import StreamingDataLoader
        except ImportError as e:
            raise DependencyError(
                "LitData is required for streaming datasets",
                package_name="litdata",
                install_command="uv add litdata",
            ) from e

        return StreamingDataLoader(
            self.test_dataset._dataset,  # type: ignore[attr-defined]
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
