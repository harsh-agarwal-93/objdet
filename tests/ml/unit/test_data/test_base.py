"""Unit tests for data base module."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from objdet.core.types import DetectionTarget
from objdet.data.base import detection_collate_fn


def _make_target(
    boxes: Tensor,
    labels: Tensor,
    **kwargs: Any,
) -> DetectionTarget:
    """Create a properly typed DetectionTarget for tests."""
    target: DetectionTarget = {
        "boxes": boxes,
        "labels": labels,
        "area": kwargs.get("area", torch.zeros(boxes.shape[0])),
        "iscrowd": kwargs.get("iscrowd", torch.zeros(boxes.shape[0], dtype=torch.int64)),
        "image_id": kwargs.get("image_id", 0),
    }
    return target


class TestDetectionCollateFn:
    """Tests for detection_collate_fn."""

    def test_collate_fn_basic(self) -> None:
        """Test basic collation of images and targets."""
        batch: list[tuple[Tensor, DetectionTarget]] = [
            (
                torch.rand(3, 224, 224),
                _make_target(
                    boxes=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    labels=torch.tensor([1]),
                ),
            ),
            (
                torch.rand(3, 224, 224),
                _make_target(
                    boxes=torch.tensor([[50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0]]),
                    labels=torch.tensor([2, 3]),
                ),
            ),
        ]

        images, targets = detection_collate_fn(batch)

        # Should return lists, not stacked tensors
        assert isinstance(images, list)
        assert isinstance(targets, list)
        assert len(images) == 2
        assert len(targets) == 2

    def test_collate_fn_preserves_image_tensors(self) -> None:
        """Test that image tensors are preserved."""
        img1 = torch.rand(3, 100, 100)
        img2 = torch.rand(3, 150, 150)

        batch: list[tuple[Tensor, DetectionTarget]] = [
            (img1, _make_target(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.int64))),
            (img2, _make_target(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.int64))),
        ]

        images, targets = detection_collate_fn(batch)

        assert torch.allclose(images[0], img1)
        assert torch.allclose(images[1], img2)

    def test_collate_fn_preserves_targets(self) -> None:
        """Test that target dictionaries are preserved."""
        target1 = _make_target(
            boxes=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            labels=torch.tensor([1]),
            area=torch.tensor([200.0]),
        )
        target2 = _make_target(
            boxes=torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
            labels=torch.tensor([2]),
            area=torch.tensor([200.0]),
        )

        batch: list[tuple[Tensor, DetectionTarget]] = [
            (torch.rand(3, 224, 224), target1),
            (torch.rand(3, 224, 224), target2),
        ]

        images, targets = detection_collate_fn(batch)

        assert torch.allclose(targets[0]["boxes"], target1["boxes"])
        assert torch.allclose(targets[1]["boxes"], target2["boxes"])

    def test_collate_fn_empty_batch(self) -> None:
        """Test collation of empty batch."""
        batch: list[tuple[Tensor, DetectionTarget]] = []

        images, targets = detection_collate_fn(batch)

        assert images == []
        assert targets == []

    def test_collate_fn_single_item(self) -> None:
        """Test collation with single item batch."""
        batch: list[tuple[Tensor, DetectionTarget]] = [
            (
                torch.rand(3, 224, 224),
                _make_target(
                    boxes=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    labels=torch.tensor([1]),
                ),
            ),
        ]

        images, targets = detection_collate_fn(batch)

        assert len(images) == 1
        assert len(targets) == 1

    def test_collate_fn_variable_box_counts(self) -> None:
        """Test collation with different number of boxes per image."""
        batch: list[tuple[Tensor, DetectionTarget]] = [
            (
                torch.rand(3, 224, 224),
                _make_target(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.int64)),
            ),
            (
                torch.rand(3, 224, 224),
                _make_target(boxes=torch.rand(5, 4), labels=torch.randint(0, 10, (5,))),
            ),
            (
                torch.rand(3, 224, 224),
                _make_target(boxes=torch.rand(2, 4), labels=torch.randint(0, 10, (2,))),
            ),
        ]

        images, targets = detection_collate_fn(batch)

        # Variable box counts should be preserved
        assert targets[0]["boxes"].shape[0] == 0
        assert targets[1]["boxes"].shape[0] == 5
        assert targets[2]["boxes"].shape[0] == 2

    def test_collate_fn_returns_tuple(self) -> None:
        """Test that collate_fn returns a tuple."""
        batch: list[tuple[Tensor, DetectionTarget]] = [
            (
                torch.rand(3, 224, 224),
                _make_target(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.int64)),
            ),
        ]

        result = detection_collate_fn(batch)

        assert isinstance(result, tuple)
        assert len(result) == 2
