"""Unit tests for class mapping module."""

from __future__ import annotations

import pytest
import torch
from objdet.data.class_mapping import (
    ClassMapper,
    create_identity_mapper,
    create_torchvision_to_yolo_mapper,
    create_yolo_to_torchvision_mapper,
)

from objdet.core.constants import ClassIndexMode
from objdet.core.exceptions import ClassMappingError


class TestClassMapper:
    """Tests for ClassMapper class."""

    @pytest.fixture
    def class_names(self) -> list[str]:
        """Sample class names."""
        return ["cat", "dog", "bird"]

    def test_yolo_to_torchvision_offset(self, class_names: list[str]) -> None:
        """Test YOLO to TorchVision mapping adds offset of 1."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        assert mapper.offset == 1
        assert mapper.needs_mapping is True

    def test_torchvision_to_yolo_offset(self, class_names: list[str]) -> None:
        """Test TorchVision to YOLO mapping has offset of -1."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.TORCHVISION,
            target_mode=ClassIndexMode.YOLO,
        )
        assert mapper.offset == -1
        assert mapper.needs_mapping is True

    def test_same_mode_no_offset(self, class_names: list[str]) -> None:
        """Test same source and target mode has no offset."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.YOLO,
        )
        assert mapper.offset == 0
        assert mapper.needs_mapping is False

    def test_map_single_label_yolo_to_tv(self, class_names: list[str]) -> None:
        """Test mapping single labels from YOLO to TorchVision."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        # YOLO 0 -> TorchVision 1
        assert mapper.map_label(0) == 1
        assert mapper.map_label(1) == 2
        assert mapper.map_label(2) == 3

    def test_map_labels_tensor(self, class_names: list[str]) -> None:
        """Test mapping tensor of labels."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        labels = torch.tensor([0, 1, 2, 0])
        mapped = mapper.map_labels(labels)
        expected = torch.tensor([1, 2, 3, 1])
        assert torch.equal(mapped, expected)

    def test_inverse_map_labels(self, class_names: list[str]) -> None:
        """Test inverse mapping from target to source."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        # TorchVision [1, 2, 3] -> YOLO [0, 1, 2]
        labels = torch.tensor([1, 2, 3])
        inverse = mapper.inverse_map_labels(labels)
        expected = torch.tensor([0, 1, 2])
        assert torch.equal(inverse, expected)

    def test_num_model_classes_torchvision(self, class_names: list[str]) -> None:
        """Test num_model_classes includes background for TorchVision."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        # 3 classes + 1 background = 4
        assert mapper.num_model_classes == 4

    def test_num_model_classes_yolo(self, class_names: list[str]) -> None:
        """Test num_model_classes is same as num_classes for YOLO."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.TORCHVISION,
            target_mode=ClassIndexMode.YOLO,
        )
        # 3 classes, no background
        assert mapper.num_model_classes == 3

    def test_get_class_name(self, class_names: list[str]) -> None:
        """Test getting class names."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        # In target format (TorchVision): 0=background, 1=cat, 2=dog
        assert mapper.get_class_name(0, in_target_format=True) == "background"
        assert mapper.get_class_name(1, in_target_format=True) == "cat"
        assert mapper.get_class_name(2, in_target_format=True) == "dog"

    def test_invalid_label_raises_error(self, class_names: list[str]) -> None:
        """Test that invalid labels raise ClassMappingError."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        # YOLO valid range is [0, 2], -1 is invalid
        with pytest.raises(ClassMappingError):
            mapper.map_label(-1)

        # 10 is also out of range
        with pytest.raises(ClassMappingError):
            mapper.map_label(10)

    def test_validate_target(self, class_names: list[str]) -> None:
        """Test target validation."""
        mapper = ClassMapper(
            class_names=class_names,
            source_mode=ClassIndexMode.YOLO,
            target_mode=ClassIndexMode.TORCHVISION,
        )
        # Valid target
        valid_target = {"labels": torch.tensor([0, 1, 2])}
        mapper.validate_target(valid_target)  # Should not raise

        # Invalid target (out of range)
        invalid_target = {"labels": torch.tensor([0, 1, 5])}
        with pytest.raises(ClassMappingError):
            mapper.validate_target(invalid_target)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_identity_mapper(self) -> None:
        """Test identity mapper creation."""
        mapper = create_identity_mapper(["a", "b"], ClassIndexMode.YOLO)
        assert mapper.offset == 0
        assert mapper.needs_mapping is False

    def test_create_yolo_to_torchvision_mapper(self) -> None:
        """Test YOLO to TorchVision mapper."""
        mapper = create_yolo_to_torchvision_mapper(["a", "b"])
        assert mapper.source_mode == ClassIndexMode.YOLO
        assert mapper.target_mode == ClassIndexMode.TORCHVISION
        assert mapper.offset == 1

    def test_create_torchvision_to_yolo_mapper(self) -> None:
        """Test TorchVision to YOLO mapper."""
        mapper = create_torchvision_to_yolo_mapper(["a", "b"])
        assert mapper.source_mode == ClassIndexMode.TORCHVISION
        assert mapper.target_mode == ClassIndexMode.YOLO
        assert mapper.offset == -1
