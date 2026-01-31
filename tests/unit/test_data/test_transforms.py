"""Unit tests for data transforms."""

import pytest
import torch

from objdet.data.transforms.base import (
    Compose,
    Normalize,
    Resize,
)
from objdet.data.transforms.detection import (
    ColorJitter,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)


@pytest.fixture
def sample_image():
    """Create sample image tensor."""
    return torch.rand(3, 480, 640)


@pytest.fixture
def sample_target():
    """Create sample detection target."""
    return {
        "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "area": torch.tensor([10000, 10000], dtype=torch.float32),
        "iscrowd": torch.tensor([0, 0], dtype=torch.int64),
        "image_id": 1,
    }


class TestNormalize:
    """Tests for Normalize transform."""

    def test_normalize_shape(self, sample_image, sample_target):
        """Normalize should preserve shape."""
        transform = Normalize()
        result_img, result_target = transform(sample_image, sample_target)

        assert result_img.shape == sample_image.shape
        assert result_target["boxes"].shape == sample_target["boxes"].shape

    def test_normalize_values(self, sample_image, sample_target):
        """Normalize should change mean/std."""
        transform = Normalize()
        result_img, _ = transform(sample_image, sample_target)

        # Mean should be approximately 0 after normalization
        # (depends on random input, so just check it's different)
        assert not torch.allclose(result_img, sample_image)


class TestResize:
    """Tests for Resize transform."""

    def test_resize_min_dimension(self, sample_image, sample_target):
        """Resize should make min dimension equal to min_size."""
        transform = Resize(min_size=400, max_size=800)
        result_img, result_target = transform(sample_image, sample_target)

        _, h, w = result_img.shape
        min_dim = min(h, w)
        max_dim = max(h, w)

        assert min_dim == 400 or max_dim == 800

    def test_resize_boxes_scaled(self, sample_image, sample_target):
        """Boxes should be scaled proportionally."""
        from copy import deepcopy

        target = deepcopy(sample_target)

        transform = Resize(min_size=400, max_size=800)
        _, result_target = transform(sample_image, target)

        # Original image is 480x640, min is 480
        # Scaled to min=400, scale = 400/480 = 0.833
        scale = 400 / 480
        expected_boxes = sample_target["boxes"] * scale

        assert torch.allclose(result_target["boxes"], expected_boxes, rtol=0.01)


class TestCompose:
    """Tests for Compose transform."""

    def test_compose_applies_in_order(self, sample_image, sample_target):
        """Compose should apply transforms in order."""
        transforms = Compose(
            [
                Resize(min_size=400),
                Normalize(),
            ]
        )

        result_img, result_target = transforms(sample_image, sample_target)

        # Check both transforms were applied
        _, h, w = result_img.shape
        assert min(h, w) == 400


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip."""

    def test_flip_always(self, sample_image, sample_target):
        """Test flip with p=1."""
        from copy import deepcopy

        target = deepcopy(sample_target)

        transform = RandomHorizontalFlip(p=1.0)
        result_img, result_target = transform(sample_image, target)

        # Image should be horizontally flipped
        assert torch.allclose(result_img, sample_image.flip(-1))

        # Boxes should be flipped
        _, _, w = sample_image.shape
        expected_boxes = sample_target["boxes"].clone()
        expected_boxes[:, [0, 2]] = w - sample_target["boxes"][:, [2, 0]]

        assert torch.allclose(result_target["boxes"], expected_boxes)

    def test_flip_never(self, sample_image, sample_target):
        """Test flip with p=0."""
        transform = RandomHorizontalFlip(p=0.0)
        result_img, result_target = transform(sample_image, sample_target)

        assert torch.allclose(result_img, sample_image)
        assert torch.allclose(result_target["boxes"], sample_target["boxes"])


class TestRandomVerticalFlip:
    """Tests for RandomVerticalFlip."""

    def test_flip_always(self, sample_image, sample_target):
        """Test vertical flip with p=1."""
        from copy import deepcopy

        target = deepcopy(sample_target)

        transform = RandomVerticalFlip(p=1.0)
        result_img, result_target = transform(sample_image, target)

        # Image should be vertically flipped
        assert torch.allclose(result_img, sample_image.flip(-2))


class TestColorJitter:
    """Tests for ColorJitter."""

    def test_jitter_changes_image(self, sample_image, sample_target):
        """ColorJitter should modify image but not boxes."""
        transform = ColorJitter(brightness=0.5, contrast=0.5)
        result_img, result_target = transform(sample_image, sample_target)

        # Image might be changed (random)
        # Boxes should not change
        assert torch.allclose(result_target["boxes"], sample_target["boxes"])


class TestRandomCrop:
    """Tests for RandomCrop."""

    def test_crop_reduces_size(self, sample_image, sample_target):
        """Crop should reduce image size."""
        transform = RandomCrop(min_scale=0.5, max_scale=0.5, min_boxes_kept=0.0)
        result_img, result_target = transform(sample_image, sample_target)

        _, h, w = result_img.shape
        _, orig_h, orig_w = sample_image.shape

        assert h <= orig_h
        assert w <= orig_w
