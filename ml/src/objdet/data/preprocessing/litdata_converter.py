"""LitData format converter for optimized data loading.

This module provides utilities for converting datasets to LitData's
optimized streaming format for faster training.

LitData stores data in an optimized format that supports:
- Fast random access
- Efficient streaming from cloud storage
- Automatic batching and prefetching

Example:
    >>> from objdet.data.preprocessing import convert_to_litdata
    >>>
    >>> convert_to_litdata(
    ...     input_dir="/data/coco",
    ...     output_dir="/data/coco_litdata",
    ...     format_name="coco",
    ...     num_workers=8,
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from objdet.core.constants import DatasetFormat
from objdet.core.exceptions import DataError, DependencyError
from objdet.core.logging import get_logger

logger = get_logger(__name__)


class _SampleEncoder:
    """Picklable sample encoder for LitData multiprocessing.

    This class must be defined at module level (not nested) to be picklable
    when using multiprocessing with spawn start method.
    """

    def __init__(self, dataset: Any) -> None:
        """Initialize the encoder with a dataset.

        Args:
            dataset: Source dataset to encode samples from.
        """
        self.dataset = dataset

    def __call__(self, idx: int) -> dict[str, Any]:
        """Encode a single sample for LitData.

        Args:
            idx: Sample index.

        Returns:
            Encoded sample dictionary.
        """
        image, target = self.dataset[idx]
        return {
            "image": image.numpy(),
            "boxes": target["boxes"].numpy(),
            "labels": target["labels"].numpy(),
            "area": target.get("area", target["boxes"]).numpy()
            if hasattr(target.get("area", None), "numpy")
            else target["boxes"][:, 2:].numpy(),  # Fallback
            "image_id": target.get("image_id", idx),
        }


class LitDataConverter:
    """Converter for creating LitData optimized datasets.

    Either provide `input_dir` to use standard directory conventions, or provide
    the format-specific path arguments directly for custom dataset structures.

    Args:
        output_dir: Output directory for LitData format.
        dataset_format: Source dataset format (coco, voc, yolo).
        input_dir: Source dataset directory (optional if format-specific paths provided).
        class_names: List of class names (required for YOLO format).
        num_workers: Number of parallel workers.
        coco_ann_file: COCO annotation file path.
        coco_images_dir: COCO images directory.
        voc_images_dir: VOC JPEGImages directory.
        voc_annotations_dir: VOC Annotations directory.
        voc_imagesets_dir: VOC ImageSets/Main directory.
        yolo_images_dir: YOLO images directory.
        yolo_labels_dir: YOLO labels directory.

    Example:
        >>> # Using standard COCO directory structure
        >>> converter = LitDataConverter(
        ...     input_dir="/data/coco",
        ...     output_dir="/data/coco_litdata",
        ...     dataset_format=DatasetFormat.COCO,
        ... )
        >>> converter.convert()

        >>> # Using custom paths (input_dir not needed)
        >>> converter = LitDataConverter(
        ...     output_dir="/data/litdata_output",
        ...     dataset_format=DatasetFormat.COCO,
        ...     coco_ann_file="/data/my_dataset/labels.json",
        ...     coco_images_dir="/data/my_dataset/photos",
        ... )
    """

    def __init__(
        self,
        output_dir: str | Path,
        dataset_format: DatasetFormat | str,
        input_dir: str | Path | None = None,
        class_names: list[str] | None = None,
        num_workers: int = 4,
        # COCO-specific paths
        coco_ann_file: str | Path | None = None,
        coco_images_dir: str | Path | None = None,
        # VOC-specific paths
        voc_images_dir: str | Path | None = None,
        voc_annotations_dir: str | Path | None = None,
        voc_imagesets_dir: str | Path | None = None,
        # YOLO-specific paths
        yolo_images_dir: str | Path | None = None,
        yolo_labels_dir: str | Path | None = None,
    ) -> None:
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.class_names = class_names

        # Store format-specific paths
        self.coco_ann_file = Path(coco_ann_file) if coco_ann_file else None
        self.coco_images_dir = Path(coco_images_dir) if coco_images_dir else None
        self.voc_images_dir = Path(voc_images_dir) if voc_images_dir else None
        self.voc_annotations_dir = Path(voc_annotations_dir) if voc_annotations_dir else None
        self.voc_imagesets_dir = Path(voc_imagesets_dir) if voc_imagesets_dir else None
        self.yolo_images_dir = Path(yolo_images_dir) if yolo_images_dir else None
        self.yolo_labels_dir = Path(yolo_labels_dir) if yolo_labels_dir else None

        if isinstance(dataset_format, str):
            dataset_format = DatasetFormat(dataset_format)
        self.dataset_format = dataset_format

        # Validate that required paths are available
        self._validate_paths()

    def _get_input_path(self, *parts: str) -> Path:
        """Safely join paths with input_dir.

        Args:
            *parts: Path parts to join.

        Returns:
            Joined path.

        Raises:
            DataError: If input_dir is not set.
        """
        if self.input_dir is None:
            raise DataError("input_dir is required when format-specific paths are not provided")
        return self.input_dir.joinpath(*parts)

    def _validate_paths(self) -> None:
        """Validate that required paths are available for the dataset format."""
        # COCO needs either input_dir OR both coco_ann_file and coco_images_dir
        if (
            self.dataset_format == DatasetFormat.COCO
            and not (self.coco_ann_file and self.coco_images_dir)
            and not self.input_dir
        ):
            raise DataError(
                "COCO format requires either 'input_dir' or both "
                "'coco_ann_file' and 'coco_images_dir'"
            )

        # VOC needs input_dir (subdirectories can be overridden individually)
        if self.dataset_format == DatasetFormat.VOC and not self.input_dir:
            raise DataError("VOC format requires 'input_dir'")

        # YOLO needs either input_dir OR both yolo_images_dir and yolo_labels_dir
        if (
            self.dataset_format == DatasetFormat.YOLO
            and not (self.yolo_images_dir and self.yolo_labels_dir)
            and not self.input_dir
        ):
            raise DataError(
                "YOLO format requires either 'input_dir' or both "
                "'yolo_images_dir' and 'yolo_labels_dir'"
            )

        # Validate that input_dir exists if provided
        if self.input_dir and not self.input_dir.exists():
            raise DataError(f"Input directory not found: {self.input_dir}")

    def convert(self, split: str = "train") -> None:
        """Convert dataset to LitData format.

        Args:
            split: Dataset split to convert.
        """
        try:
            from litdata import optimize
        except ImportError as e:
            raise DependencyError(
                "LitData is required for conversion",
                package_name="litdata",
                install_command="uv add litdata",
            ) from e

        logger.info(
            f"Converting {self.dataset_format.value} dataset to LitData format",
            input_dir=str(self.input_dir) if self.input_dir else "N/A",
            output_dir=str(self.output_dir),
            split=split,
        )

        # Create output directory
        output_split_dir = self.output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # Get source dataset
        dataset = self._create_source_dataset(split)

        # Create picklable encoder (must be module-level class, not nested function)
        encoder = _SampleEncoder(dataset)

        # Run optimization
        optimize(
            fn=encoder,
            inputs=list(range(len(dataset))),
            output_dir=str(output_split_dir),
            num_workers=self.num_workers,
            chunk_bytes="64MB",
        )

        logger.info(f"Conversion complete: {len(dataset)} samples written to {output_split_dir}")

    def _create_source_dataset(self, split: str) -> Any:
        """Create source dataset based on format.

        Args:
            split: Dataset split.

        Returns:
            Dataset instance.
        """
        if self.dataset_format == DatasetFormat.COCO:
            from objdet.data.formats.coco import COCODataset

            # Use overrides if provided, otherwise use standard COCO structure
            ann_file = (
                self.coco_ann_file
                if self.coco_ann_file
                else self._get_input_path(f"annotations/instances_{split}2017.json")
            )
            images_dir = (
                self.coco_images_dir
                if self.coco_images_dir
                else self._get_input_path(f"{split}2017")
            )

            return COCODataset(
                data_dir=images_dir,
                ann_file=ann_file,
            )

        elif self.dataset_format == DatasetFormat.VOC:
            from objdet.data.formats.voc import VOCDataset

            # VOC uses data_dir with internal structure, but we can override
            # individual subdirectories via the dataset if needed
            return VOCDataset(
                data_dir=self.input_dir,  # type: ignore[arg-type] # Checked in _validate_paths
                split=split,
                class_names=self.class_names,
                images_dir=self.voc_images_dir,
                annotations_dir=self.voc_annotations_dir,
                imagesets_dir=self.voc_imagesets_dir,
            )

        elif self.dataset_format == DatasetFormat.YOLO:
            from objdet.data.formats.yolo import YOLODataset

            if not self.class_names:
                raise DataError("class_names is required for YOLO format conversion")

            # Use overrides if provided, otherwise use standard YOLO structure
            images_dir = (
                self.yolo_images_dir
                if self.yolo_images_dir
                else self._get_input_path("images", split)
            )
            labels_dir = (
                self.yolo_labels_dir
                if self.yolo_labels_dir
                else self._get_input_path("labels", split)
            )

            return YOLODataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                class_names=self.class_names,
            )

        else:
            raise DataError(f"Unsupported format: {self.dataset_format}")


def convert_to_litdata(
    output_dir: str | Path,
    format_name: str,
    input_dir: str | Path | None = None,
    num_workers: int = 4,
    class_names: list[str] | None = None,
    splits: list[str] | None = None,
    # COCO-specific paths
    coco_ann_file: str | Path | None = None,
    coco_images_dir: str | Path | None = None,
    # VOC-specific paths
    voc_images_dir: str | Path | None = None,
    voc_annotations_dir: str | Path | None = None,
    voc_imagesets_dir: str | Path | None = None,
    # YOLO-specific paths
    yolo_images_dir: str | Path | None = None,
    yolo_labels_dir: str | Path | None = None,
) -> None:
    """Convert dataset to LitData format.

    This is the main entry point for the CLI preprocess command.

    Either provide `input_dir` to use standard directory conventions, or provide
    the format-specific path arguments directly for custom dataset structures.

    Args:
        output_dir: Output directory for LitData format.
        format_name: Source format ("coco", "voc", "yolo").
        input_dir: Source dataset directory (optional if format-specific paths provided).
        num_workers: Number of parallel workers.
        class_names: Class names (required for YOLO).
        splits: List of splits to convert (default: ["train", "val"]).
        coco_ann_file: COCO annotation file path.
        coco_images_dir: COCO images directory.
        voc_images_dir: VOC JPEGImages directory.
        voc_annotations_dir: VOC Annotations directory.
        voc_imagesets_dir: VOC ImageSets/Main directory.
        yolo_images_dir: YOLO images directory.
        yolo_labels_dir: YOLO labels directory.

    Example:
        >>> # Using standard directory structure
        >>> convert_to_litdata(
        ...     input_dir="/data/coco",
        ...     output_dir="/data/coco_litdata",
        ...     format_name="coco",
        ... )

        >>> # Using custom paths (input_dir not needed)
        >>> convert_to_litdata(
        ...     output_dir="/data/litdata_output",
        ...     format_name="coco",
        ...     coco_ann_file="/data/my_coco/train_annotations.json",
        ...     coco_images_dir="/data/my_coco/images",
        ...     splits=["train"],
        ... )
    """
    splits = splits or ["train", "val"]

    converter = LitDataConverter(
        output_dir=output_dir,
        dataset_format=format_name,
        input_dir=input_dir,
        class_names=class_names,
        num_workers=num_workers,
        coco_ann_file=coco_ann_file,
        coco_images_dir=coco_images_dir,
        voc_images_dir=voc_images_dir,
        voc_annotations_dir=voc_annotations_dir,
        voc_imagesets_dir=voc_imagesets_dir,
        yolo_images_dir=yolo_images_dir,
        yolo_labels_dir=yolo_labels_dir,
    )

    for split in splits:
        try:
            converter.convert(split=split)
        except Exception as e:
            logger.error(f"Failed to convert split '{split}': {e}")
            raise
