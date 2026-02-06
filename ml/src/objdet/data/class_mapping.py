"""Class index mapping between different model formats.

This module provides explicit handling of class indices between
YOLO format (no background class) and TorchVision format (background at 0).

CRITICAL: This is a common source of bugs in detection pipelines.
The ClassMapper makes the mapping explicit and avoids off-by-one errors.

Example:
    >>> # Dataset has classes: ["cat", "dog", "bird"]
    >>> mapper = ClassMapper(
    ...     class_names=["cat", "dog", "bird"],
    ...     source_mode=ClassIndexMode.YOLO,
    ...     target_mode=ClassIndexMode.TORCHVISION,
    ... )
    >>>
    >>> # YOLO class 0 (cat) -> TorchVision class 1 (background at 0)
    >>> mapper.map_label(0)
    1
"""

from __future__ import annotations

from torch import Tensor

from objdet.core.constants import ClassIndexMode
from objdet.core.exceptions import ClassMappingError
from objdet.core.logging import get_logger

logger = get_logger(__name__)


class ClassMapper:
    """Explicit class index mapper between formats.

    This class handles the conversion of class indices between:
    - YOLO format: Classes are 0-indexed, no background class
    - TorchVision format: Background is class 0, user classes start at 1

    Args:
        class_names: List of class names (NOT including background).
        source_mode: The class index mode of the source data/predictions.
        target_mode: The class index mode expected by the target model/format.

    Attributes:
        class_names: List of user class names.
        source_mode: Source class index mode.
        target_mode: Target class index mode.
        offset: The offset to add/subtract when mapping.

    Example:
        >>> mapper = ClassMapper(
        ...     class_names=["person", "car"],
        ...     source_mode=ClassIndexMode.YOLO,
        ...     target_mode=ClassIndexMode.TORCHVISION,
        ... )
        >>> # YOLO labels [0, 1] become TorchVision labels [1, 2]
        >>> mapper.map_labels(torch.tensor([0, 1]))
        tensor([1, 2])
    """

    def __init__(
        self,
        class_names: list[str],
        source_mode: ClassIndexMode,
        target_mode: ClassIndexMode,
    ) -> None:
        self.class_names = class_names
        self.source_mode = source_mode
        self.target_mode = target_mode

        # Calculate offset
        # YOLO -> TorchVision: add 1 (shift for background)
        # TorchVision -> YOLO: subtract 1
        # Same mode: no change
        source_offset = source_mode.user_class_offset()
        target_offset = target_mode.user_class_offset()
        self.offset = target_offset - source_offset

        if self.offset != 0:
            logger.debug(
                f"ClassMapper: {source_mode.value} -> {target_mode.value}, offset={self.offset}"
            )

    @property
    def needs_mapping(self) -> bool:
        """Check if mapping is needed (modes are different).

        Returns:
            True if source and target modes differ.
        """
        return self.offset != 0

    @property
    def num_classes(self) -> int:
        """Get number of user classes (not including background).

        Returns:
            Number of user classes.
        """
        return len(self.class_names)

    @property
    def num_model_classes(self) -> int:
        """Get number of classes expected by target model.

        For TorchVision target: num_classes + 1 (including background)
        For YOLO target: num_classes

        Returns:
            Number of classes for the target model.
        """
        if self.target_mode == ClassIndexMode.TORCHVISION:
            return self.num_classes + 1
        return self.num_classes

    def map_label(self, label: int) -> int:
        """Map a single label from source to target format.

        Args:
            label: Class label in source format.

        Returns:
            Class label in target format.

        Raises:
            ClassMappingError: If label is out of valid range.
        """
        # Validate source label
        max_source_label = self.num_classes - 1 + self.source_mode.user_class_offset()
        min_source_label = self.source_mode.user_class_offset()

        if label < min_source_label or label > max_source_label:
            raise ClassMappingError(
                f"Invalid label {label} for {self.source_mode.value} mode. "
                f"Expected range [{min_source_label}, {max_source_label}]."
            )

        return label + self.offset

    def map_labels(self, labels: Tensor) -> Tensor:
        """Map a tensor of labels from source to target format.

        Args:
            labels: Tensor of class labels in source format.

        Returns:
            Tensor of class labels in target format.
        """
        if self.offset == 0:
            return labels

        return labels + self.offset

    def inverse_map_label(self, label: int) -> int:
        """Map a label from target back to source format.

        Args:
            label: Class label in target format.

        Returns:
            Class label in source format.
        """
        return label - self.offset

    def inverse_map_labels(self, labels: Tensor) -> Tensor:
        """Map a tensor of labels from target back to source format.

        Args:
            labels: Tensor of class labels in target format.

        Returns:
            Tensor of class labels in source format.
        """
        if self.offset == 0:
            return labels

        return labels - self.offset

    def get_class_name(self, label: int, in_target_format: bool = True) -> str:
        """Get class name for a label.

        Args:
            label: Class label.
            in_target_format: If True, label is in target format.
                If False, label is in source format.

        Returns:
            Class name string.
        """
        # Convert to 0-indexed user class
        if in_target_format:
            user_idx = label - self.target_mode.user_class_offset()
        else:
            user_idx = label - self.source_mode.user_class_offset()

        # Handle background
        if user_idx < 0:
            return "background"

        if 0 <= user_idx < len(self.class_names):
            return self.class_names[user_idx]

        return f"unknown_{user_idx}"

    def validate_target(self, target: dict[str, Tensor]) -> None:
        """Validate that target labels are in expected source format.

        Args:
            target: Detection target dictionary with 'labels' key.

        Raises:
            ClassMappingError: If labels are out of valid range.
        """
        if "labels" not in target:
            return

        labels = target["labels"]
        if labels.numel() == 0:
            return

        min_label = labels.min().item()
        max_label = labels.max().item()

        expected_min = self.source_mode.user_class_offset()
        expected_max = self.num_classes - 1 + expected_min

        if min_label < expected_min or max_label > expected_max:
            raise ClassMappingError(
                f"Labels out of range for {self.source_mode.value} mode. "
                f"Got range [{min_label}, {max_label}], "
                f"expected [{expected_min}, {expected_max}]."
            )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClassMapper("
            f"source={self.source_mode.value}, "
            f"target={self.target_mode.value}, "
            f"num_classes={self.num_classes}, "
            f"offset={self.offset})"
        )


def create_identity_mapper(
    class_names: list[str],
    mode: ClassIndexMode,
) -> ClassMapper:
    """Create a mapper that doesn't change indices (same source and target).

    Args:
        class_names: List of class names.
        mode: The class index mode.

    Returns:
        ClassMapper with source and target set to the same mode.
    """
    return ClassMapper(
        class_names=class_names,
        source_mode=mode,
        target_mode=mode,
    )


def create_yolo_to_torchvision_mapper(class_names: list[str]) -> ClassMapper:
    """Create a mapper from YOLO to TorchVision format.

    Args:
        class_names: List of class names (not including background).

    Returns:
        ClassMapper that adds 1 to all labels.
    """
    return ClassMapper(
        class_names=class_names,
        source_mode=ClassIndexMode.YOLO,
        target_mode=ClassIndexMode.TORCHVISION,
    )


def create_torchvision_to_yolo_mapper(class_names: list[str]) -> ClassMapper:
    """Create a mapper from TorchVision to YOLO format.

    Args:
        class_names: List of class names (not including background).

    Returns:
        ClassMapper that subtracts 1 from all labels.
    """
    return ClassMapper(
        class_names=class_names,
        source_mode=ClassIndexMode.TORCHVISION,
        target_mode=ClassIndexMode.YOLO,
    )
