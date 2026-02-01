"""Unit tests for SAHI wrapper (SlicedInference)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor


class TestSlicedInference:
    """Tests for SlicedInference class."""

    @pytest.fixture
    def mock_predictor(self) -> MagicMock:
        """Create a mock predictor that returns dummy predictions."""
        predictor = MagicMock()

        def mock_predict(image: Tensor) -> dict:
            # Return simple prediction with one box
            return {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }

        predictor.predict = mock_predict
        return predictor

    @pytest.fixture
    def mock_predictor_empty(self) -> MagicMock:
        """Create a mock predictor that returns no predictions."""
        predictor = MagicMock()

        def mock_predict(image: Tensor) -> dict:
            return {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            }

        predictor.predict = mock_predict
        return predictor

    def test_init(self, mock_predictor: MagicMock) -> None:
        """Test initialization with default parameters."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(predictor=mock_predictor)

        assert sahi.slice_height == 640
        assert sahi.slice_width == 640
        assert sahi.overlap_ratio == 0.2
        assert sahi.merge_method == "nms"
        assert sahi.nms_threshold == 0.5
        assert sahi.include_full_image is True

    def test_init_custom_params(self, mock_predictor: MagicMock) -> None:
        """Test initialization with custom parameters."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            slice_height=512,
            slice_width=512,
            overlap_ratio=0.3,
            merge_method="wbf",
            nms_threshold=0.4,
            include_full_image=False,
        )

        assert sahi.slice_height == 512
        assert sahi.slice_width == 512
        assert sahi.overlap_ratio == 0.3
        assert sahi.merge_method == "wbf"
        assert sahi.nms_threshold == 0.4
        assert sahi.include_full_image is False

    def test_get_slice_positions_basic(self, mock_predictor: MagicMock) -> None:
        """Test slice position calculation for basic case."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            slice_height=100,
            slice_width=100,
            overlap_ratio=0.0,
        )

        slices = sahi._get_slice_positions(img_height=200, img_width=200)

        # Should have 4 slices (2x2 grid)
        assert len(slices) == 4
        # Each slice should be 100x100
        for x1, y1, x2, y2 in slices:
            assert x2 - x1 == 100
            assert y2 - y1 == 100

    def test_get_slice_positions_with_overlap(self, mock_predictor: MagicMock) -> None:
        """Test slice position calculation with overlap."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            slice_height=100,
            slice_width=100,
            overlap_ratio=0.5,  # 50% overlap
        )

        slices = sahi._get_slice_positions(img_height=200, img_width=200)

        # With 50% overlap and 100px slices on 200px image:
        # step = 50, so positions are 0, 50, 100, 150
        # More slices due to overlap
        assert len(slices) > 4

    def test_get_slice_positions_small_image(self, mock_predictor: MagicMock) -> None:
        """Test slice positions when image is smaller than slice size."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            slice_height=640,
            slice_width=640,
            overlap_ratio=0.2,
        )

        slices = sahi._get_slice_positions(img_height=320, img_width=480)

        # Should have at least 1 slice
        assert len(slices) >= 1
        # Slice should cover entire image
        x1, y1, x2, y2 = slices[0]
        assert x2 <= 640
        assert y2 <= 640

    def test_merge_nms(self, mock_predictor: MagicMock) -> None:
        """Test NMS merge method."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            merge_method="nms",
            nms_threshold=0.5,
        )

        # Create overlapping boxes
        boxes = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [5.0, 5.0, 105.0, 105.0],  # Overlaps with first
                [200.0, 200.0, 300.0, 300.0],  # No overlap
            ]
        )
        labels = torch.tensor([1, 1, 2])
        scores = torch.tensor([0.9, 0.8, 0.95])

        result = sahi._merge_nms(boxes, labels, scores)

        # NMS should keep high-scoring boxes and remove overlaps for same class
        assert "boxes" in result
        assert "labels" in result
        assert "scores" in result
        assert len(result["boxes"]) <= 3

    def test_predict_with_tensor(self, mock_predictor: MagicMock) -> None:
        """Test prediction with tensor input."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            slice_height=200,
            slice_width=200,
            overlap_ratio=0.0,
            include_full_image=False,
        )

        # Create test image tensor
        image = torch.randn(3, 400, 400)

        result = sahi.predict(image)

        assert "boxes" in result
        assert "labels" in result
        assert "scores" in result

    def test_predict_empty_results(self, mock_predictor_empty: MagicMock) -> None:
        """Test prediction when no detections are made."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor_empty,
            slice_height=200,
            slice_width=200,
            include_full_image=False,
        )

        image = torch.randn(3, 400, 400)
        result = sahi.predict(image)

        assert result["boxes"].shape == (0, 4)
        assert result["labels"].shape == (0,)
        assert result["scores"].shape == (0,)

    def test_merge_method_invalid_raises(self, mock_predictor: MagicMock) -> None:
        """Test that invalid merge method raises error."""
        from objdet.inference.sahi_wrapper import SlicedInference

        sahi = SlicedInference(
            predictor=mock_predictor,
            merge_method="invalid",
        )

        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        labels = torch.tensor([1])
        scores = torch.tensor([0.9])

        with pytest.raises(ValueError, match="Unknown merge method"):
            sahi._merge_predictions(boxes, labels, scores, 100, 100)

    def test_predict_includes_full_image(self, mock_predictor: MagicMock) -> None:
        """Test that full image prediction is included when enabled."""
        from objdet.inference.sahi_wrapper import SlicedInference

        call_count = [0]
        original_predict = mock_predictor.predict

        def counting_predict(image: Tensor) -> dict:
            call_count[0] += 1
            return original_predict(image)

        mock_predictor.predict = counting_predict

        sahi = SlicedInference(
            predictor=mock_predictor,
            slice_height=200,
            slice_width=200,
            overlap_ratio=0.0,
            include_full_image=True,
        )

        image = torch.randn(3, 400, 400)
        sahi.predict(image)

        # Should have more calls than just slices due to full image
        # 4 slices (2x2) + 1 full image = 5 calls
        assert call_count[0] >= 5
