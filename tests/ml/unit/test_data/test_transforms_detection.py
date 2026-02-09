from typing import cast
from unittest.mock import patch

import torch

from objdet.core.types import DetectionTarget
from objdet.data.transforms.base import Compose
from objdet.data.transforms.detection import (
    ColorJitter,
    DetectionTransform,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    get_train_transforms,
    get_val_transforms,
)


def test_clip_boxes() -> None:
    """Test box clipping to image boundaries."""
    boxes = torch.tensor([[-10.0, -10.0, 110.0, 110.0], [10.0, 10.0, 50.0, 50.0]])
    clipped = DetectionTransform.clip_boxes(boxes, height=100, width=100)

    expected = torch.tensor([[0.0, 0.0, 100.0, 100.0], [10.0, 10.0, 50.0, 50.0]])
    assert torch.allclose(clipped, expected)


def test_remove_small_boxes() -> None:
    """Test removal of boxes smaller than threshold."""
    target = cast(
        "DetectionTarget",
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 2.0, 2.0]]),
            "labels": torch.tensor([1, 2]),
        },
    )
    filtered = DetectionTransform.remove_small_boxes(target, min_size=1.0)

    assert len(filtered["boxes"]) == 1
    assert filtered["labels"][0] == 2

    # Empty case
    empty_target = cast(
        "DetectionTarget",
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
    )
    res = DetectionTransform.remove_small_boxes(empty_target)
    assert len(res["boxes"]) == 0


def test_random_horizontal_flip() -> None:
    """Test random horizontal flip."""
    image = torch.zeros(3, 100, 100)
    target = cast(
        "DetectionTarget",
        {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
    )

    # Test with p=1.0 (always flip)
    flip = RandomHorizontalFlip(p=1.0)
    with patch("random.random", return_value=0.0):
        _, target_f = flip(image, target)
        expected_box = torch.tensor([[80.0, 10.0, 90.0, 20.0]])
        assert torch.allclose(target_f["boxes"], expected_box)

    # Test with p=0.0 (never flip)
    no_flip = RandomHorizontalFlip(p=0.0)
    _, target_nf = no_flip(image, target)
    assert torch.allclose(target_nf["boxes"], target["boxes"])


def test_random_vertical_flip() -> None:
    """Test random vertical flip."""
    image = torch.zeros(3, 100, 100)
    target = cast(
        "DetectionTarget",
        {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
    )

    # Always flip
    flip = RandomVerticalFlip(p=1.0)
    with patch("random.random", return_value=0.0):
        _, target_f = flip(image, target)
        expected_box = torch.tensor([[10.0, 80.0, 20.0, 90.0]])
        assert torch.allclose(target_f["boxes"], expected_box)


def test_color_jitter() -> None:
    """Test color jitter augmentation."""
    image = torch.ones(3, 100, 100) * 0.5
    target = cast(
        "DetectionTarget",
        {"boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]), "labels": torch.tensor([1])},
    )

    jitter = ColorJitter(brightness=0.5, contrast=0.5)
    img_j, _ = jitter(image, target)

    assert img_j.shape == image.shape
    # Values should be different but clamped
    assert torch.all(img_j >= 0) and torch.all(img_j <= 1)


def test_random_crop_success() -> None:
    """Test successful random crop."""
    image = torch.randn(3, 100, 100)
    target = cast(
        "DetectionTarget",
        {
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([1]),
        },
    )

    # Force a crop that includes the box
    crop = RandomCrop(min_scale=0.5, max_scale=0.5, min_boxes_kept=0.1)
    with (
        patch("random.uniform", return_value=0.5),
        patch("random.randint", return_value=30),
    ):  # top=30, left=30
        img_c, target_c = crop(image, target)

        assert img_c.shape == (3, 50, 50)
        assert len(target_c["boxes"]) == 1
        # Adjusted box: [40-30, 40-30, 60-30, 60-30] -> [10, 10, 30, 30]
        assert torch.allclose(target_c["boxes"], torch.tensor([[10.0, 10.0, 30.0, 30.0]]))


def test_random_crop_fail_ratio() -> None:
    """Test random crop that is aborted due to low keep ratio."""
    image = torch.randn(3, 100, 100)
    target = cast(
        "DetectionTarget",
        {
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            "labels": torch.tensor([1]),
        },
    )

    # Force a crop that misses the box
    crop = RandomCrop(min_scale=0.5, max_scale=0.5, min_boxes_kept=0.9)
    with (
        patch("random.uniform", return_value=0.5),
        patch("random.randint", return_value=50),
    ):  # top=50, left=50
        img_c, target_c = crop(image, target)

        # Should return original image and target
        assert img_c.shape == (3, 100, 100)
        assert torch.allclose(target_c["boxes"], target["boxes"])


def test_get_transforms() -> None:
    """Test transform factory functions."""
    train_tf = cast("Compose", get_train_transforms(use_augmentation=True))
    assert len(train_tf.transforms) > 2

    val_tf = cast("Compose", get_val_transforms())
    assert len(val_tf.transforms) == 2
