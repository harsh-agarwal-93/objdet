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

from objdet.core.constants import ClassIndexMode, DatasetFormat
from objdet.core.exceptions import DataError, DependencyError
from objdet.core.logging import get_logger
from objdet.data.base import BaseDataModule, detection_collate_fn
from objdet.data.registry import DATAMODULE_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

    from objdet.core.types import DetectionTarget

logger = get_logger(__name__)


def _detection_transform(sample: dict[str, Any]) -> tuple[Tensor, DetectionTarget]:
    """Transform raw LitData sample to detection format.

    This function converts numpy arrays from LitData's StreamingDataset
    to PyTorch tensors in the detection format expected by models.

    Args:
        sample: Raw sample from StreamingDataset containing numpy arrays.

    Returns:
        Tuple of (image_tensor, target_dict).
    """
    # Convert numpy arrays to tensors (copy to make writable)
    image = torch.from_numpy(sample["image"].copy()).float()
    target: DetectionTarget = {
        "boxes": torch.from_numpy(sample["boxes"].copy()).float(),
        "labels": torch.from_numpy(sample["labels"].copy()).long(),
        "area": torch.from_numpy(sample["area"].copy()).float(),
        "image_id": sample["image_id"],
    }
    return image, target


class DetectionStreamingDataset:
    """StreamingDataset wrapper for object detection tasks.

    This class wraps LitData's StreamingDataset and applies the detection
    transform to convert samples to (image_tensor, target_dict) tuples.

    Inherits optimized streaming capabilities from litdata:
    - Efficient chunked data loading
    - Automatic prefetching and caching
    - Support for distributed training
    - Cloud storage streaming

    Args:
        input_dir: Directory containing LitData optimized chunks.
        shuffle: Whether to shuffle the data.
        drop_last: Whether to drop the last incomplete batch.
        transforms: Optional additional transform to apply after detection transform.

    Attributes:
        input_dir: Path to the data directory.

    Example:
        >>> dataset = DetectionStreamingDataset(
        ...     input_dir="/data/coco_litdata/train",
        ...     shuffle=True,
        ... )
    """

    def __new__(
        cls,
        input_dir: str | Path,
        shuffle: bool = False,
        drop_last: bool = False,
        transforms: Callable | None = None,
    ) -> Any:
        """Create a StreamingDataset with detection transforms.

        Returns the underlying StreamingDataset directly to ensure compatibility
        with StreamingDataLoader which requires the exact StreamingDataset type.
        """
        try:
            from litdata import StreamingDataset
        except ImportError as e:
            raise DependencyError(
                "LitData is required for streaming datasets",
                package_name="litdata",
                install_command="uv add litdata",
            ) from e

        data_dir = Path(input_dir)
        if not data_dir.exists():
            raise DataError(f"Data directory not found: {data_dir}")

        # Build transform chain: detection transform + optional user transform
        def combined_transform(sample: dict[str, Any]) -> tuple[Tensor, DetectionTarget]:
            image, target = _detection_transform(sample)
            if transforms is not None:
                image, target = transforms(image, target)
            return image, target

        # Create StreamingDataset with detection transform
        dataset = StreamingDataset(
            input_dir=str(data_dir),
            shuffle=shuffle,
            drop_last=drop_last,
            transform=combined_transform,
        )

        logger.info(
            f"Created streaming dataset from {data_dir}",
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return dataset


def create_streaming_dataloader(
    dataset: Any,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Any:
    """Create a StreamingDataLoader for detection tasks.

    Uses LitData's StreamingDataLoader with a custom collate function
    for handling variable-length detection targets.

    Args:
        dataset: A StreamingDataset instance.
        batch_size: Batch size for the dataloader.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        StreamingDataLoader configured for detection tasks.
    """
    try:
        from litdata import StreamingDataLoader
    except ImportError as e:
        raise DependencyError(
            "LitData is required for streaming datasets",
            package_name="litdata",
            install_command="uv add litdata",
        ) from e

    return StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=detection_collate_fn,
    )


@DATAMODULE_REGISTRY.register("litdata", aliases=["streaming"])
class LitDataDataModule(BaseDataModule):
    """Lightning DataModule for LitData optimized datasets.

    This DataModule uses LitData's StreamingDataset and StreamingDataLoader
    for efficient data loading from preprocessed datasets.

    Features:
    - Uses native LitData streaming for optimal performance
    - Custom collate function for variable-length detection targets
    - Automatic transform chain for numpy-to-tensor conversion

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
        kwargs.setdefault("dataset_format", DatasetFormat.LITDATA)

        super().__init__(
            data_dir=data_dir,
            class_names=class_names,
            class_index_mode=class_index_mode,
            **kwargs,
        )

        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
        self.test_subdir = test_subdir

        # Store streaming datasets separately (they are the native StreamingDataset)
        self._train_streaming: Any = None
        self._val_streaming: Any = None
        self._test_streaming: Any = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: One of "fit", "validate", "test", or "predict".
        """
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / self.train_subdir
            if train_dir.exists():
                self._train_streaming = DetectionStreamingDataset(
                    input_dir=train_dir,
                    shuffle=True,
                    drop_last=True,
                    transforms=self.train_transforms,
                )
                # Also set train_dataset for compatibility with base class
                self.train_dataset = self._train_streaming  # type: ignore[assignment]

            val_dir = self.data_dir / self.val_subdir
            if val_dir.exists():
                self._val_streaming = DetectionStreamingDataset(
                    input_dir=val_dir,
                    shuffle=False,
                    transforms=self.val_transforms,
                )
                self.val_dataset = self._val_streaming  # type: ignore[assignment]

        if stage == "validate" and self._val_streaming is None:
            val_dir = self.data_dir / self.val_subdir
            if val_dir.exists():
                self._val_streaming = DetectionStreamingDataset(
                    input_dir=val_dir,
                    shuffle=False,
                    transforms=self.val_transforms,
                )
                self.val_dataset = self._val_streaming  # type: ignore[assignment]

        if stage == "test":
            test_dir = self.data_dir / self.test_subdir
            if test_dir.exists():
                self._test_streaming = DetectionStreamingDataset(
                    input_dir=test_dir,
                    shuffle=False,
                    transforms=self.test_transforms,
                )
                self.test_dataset = self._test_streaming  # type: ignore[assignment]

    def train_dataloader(self) -> Any:
        """Create training DataLoader using LitData's StreamingDataLoader.

        Returns:
            StreamingDataLoader for training data with shuffling enabled.
        """
        if self._train_streaming is None:
            raise RuntimeError("train_dataset not set. Call setup('fit') first.")

        return create_streaming_dataloader(
            dataset=self._train_streaming,
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
        if self._val_streaming is None:
            raise RuntimeError("val_dataset not set. Call setup('fit' or 'validate') first.")

        return create_streaming_dataloader(
            dataset=self._val_streaming,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> Any:
        """Create test DataLoader using LitData's StreamingDataLoader.

        Returns:
            StreamingDataLoader for test data without shuffling.
        """
        if self._test_streaming is None:
            raise RuntimeError("test_dataset not set. Call setup('test') first.")

        return create_streaming_dataloader(
            dataset=self._test_streaming,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
