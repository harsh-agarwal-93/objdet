"""Unit tests for ensemble models using mocks."""

from __future__ import annotations

import numpy as np

from tests.unit.mocks.mock_ensemble_boxes import (  # type: ignore
    mock_nms,
    mock_soft_nms,
    mock_weighted_boxes_fusion,
)


class TestMockNMS:
    """Tests for mock NMS function."""

    def test_nms_with_single_model(self) -> None:
        """Test NMS with predictions from a single model."""
        boxes = [np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])]
        scores = [np.array([0.9, 0.8])]
        labels = [np.array([0.0, 1.0])]

        result_boxes, result_scores, result_labels = mock_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
        )

        assert len(result_boxes) > 0
        assert len(result_scores) > 0
        assert len(result_labels) > 0

    def test_nms_with_multiple_models(self) -> None:
        """Test NMS with predictions from multiple models."""
        boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3]]),
            np.array([[0.12, 0.12, 0.32, 0.32]]),
            np.array([[0.5, 0.5, 0.7, 0.7]]),
        ]
        scores = [
            np.array([0.9]),
            np.array([0.85]),
            np.array([0.8]),
        ]
        labels = [
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        ]

        result_boxes, result_scores, result_labels = mock_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
        )

        # Should have at most one box per class
        assert len(np.unique(result_labels)) == len(result_labels)

    def test_nms_empty_input(self) -> None:
        """Test NMS with empty input."""
        boxes = [np.empty((0, 4))]
        scores = [np.empty(0)]
        labels = [np.empty(0)]

        result_boxes, result_scores, result_labels = mock_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
        )

        assert len(result_boxes) == 0
        assert len(result_scores) == 0
        assert len(result_labels) == 0

    def test_nms_filters_low_scores(self) -> None:
        """Test that NMS filters boxes below threshold."""
        boxes = [np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])]
        scores = [np.array([0.9, 0.1])]
        labels = [np.array([0.0, 1.0])]

        result_boxes, result_scores, result_labels = mock_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
            skip_box_thr=0.5,
        )

        # Only high score box should remain
        assert all(s >= 0.5 for s in result_scores)


class TestMockSoftNMS:
    """Tests for mock Soft-NMS function."""

    def test_soft_nms_basic(self) -> None:
        """Test basic Soft-NMS functionality."""
        boxes = [np.array([[0.1, 0.1, 0.3, 0.3]])]
        scores = [np.array([0.9])]
        labels = [np.array([0.0])]

        result_boxes, result_scores, result_labels = mock_soft_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
        )

        assert len(result_boxes) == 1


class TestMockWBF:
    """Tests for mock Weighted Box Fusion function."""

    def test_wbf_with_single_model(self) -> None:
        """Test WBF with predictions from a single model."""
        boxes = [np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])]
        scores = [np.array([0.9, 0.8])]
        labels = [np.array([0.0, 1.0])]

        result_boxes, result_scores, result_labels = mock_weighted_boxes_fusion(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.55,
        )

        assert len(result_boxes) > 0

    def test_wbf_with_weights(self) -> None:
        """Test WBF with model weights."""
        boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3]]),
            np.array([[0.12, 0.12, 0.32, 0.32]]),
        ]
        scores = [
            np.array([0.9]),
            np.array([0.85]),
        ]
        labels = [
            np.array([0.0]),
            np.array([0.0]),
        ]
        weights = [0.6, 0.4]

        result_boxes, result_scores, result_labels = mock_weighted_boxes_fusion(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            weights=weights,
            iou_thr=0.55,
        )

        # Should fuse overlapping boxes
        assert len(result_boxes) == 1

    def test_wbf_empty_input(self) -> None:
        """Test WBF with empty input."""
        boxes = [np.empty((0, 4))]
        scores = [np.empty(0)]
        labels = [np.empty(0)]

        result_boxes, result_scores, result_labels = mock_weighted_boxes_fusion(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.55,
        )

        assert len(result_boxes) == 0

    def test_wbf_multiple_classes(self) -> None:
        """Test WBF with multiple classes."""
        boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]),
        ]
        scores = [np.array([0.9, 0.85])]
        labels = [np.array([0.0, 1.0])]

        result_boxes, result_scores, result_labels = mock_weighted_boxes_fusion(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.55,
        )

        # Should have one box per class
        assert len(np.unique(result_labels)) == 2


class TestEnsemblePredictionFormat:
    """Tests for ensemble prediction output format."""

    def test_prediction_has_required_keys(self) -> None:
        """Test that mock predictions have required format."""
        boxes = [np.array([[0.1, 0.1, 0.3, 0.3]])]
        scores = [np.array([0.9])]
        labels = [np.array([0.0])]

        result_boxes, result_scores, result_labels = mock_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
        )

        # Check shapes
        assert result_boxes.ndim == 2
        assert result_boxes.shape[1] == 4
        assert result_scores.ndim == 1
        assert result_labels.ndim == 1

    def test_multiple_boxes_format(self) -> None:
        """Test format with multiple output boxes."""
        boxes = [np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7], [0.8, 0.8, 0.9, 0.9]])]
        scores = [np.array([0.9, 0.8, 0.7])]
        labels = [np.array([0.0, 1.0, 2.0])]

        result_boxes, result_scores, result_labels = mock_nms(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            iou_thr=0.5,
        )

        # All outputs should have consistent lengths
        assert len(result_boxes) == len(result_scores) == len(result_labels)


class TestWeightNormalization:
    """Tests for model weight handling in ensemble."""

    def test_weights_are_normalized(self) -> None:
        """Test that weights are normalized in WBF."""
        boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3]]),
            np.array([[0.1, 0.1, 0.3, 0.3]]),
        ]
        scores = [
            np.array([0.9]),
            np.array([0.9]),
        ]
        labels = [
            np.array([0.0]),
            np.array([0.0]),
        ]

        # Weights that don't sum to 1
        weights = [2.0, 3.0]

        result_boxes, result_scores, result_labels = mock_weighted_boxes_fusion(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            weights=weights,
            iou_thr=0.55,
        )

        # Should still produce valid output
        assert len(result_boxes) == 1
        # Score should be normalized (not > 1)
        assert all(s <= 1.0 for s in result_scores)

    def test_equal_weights_default(self) -> None:
        """Test that equal weights are used by default."""
        boxes = [
            np.array([[0.1, 0.1, 0.3, 0.3]]),
            np.array([[0.1, 0.1, 0.3, 0.3]]),
        ]
        scores = [
            np.array([0.9]),
            np.array([0.8]),
        ]
        labels = [
            np.array([0.0]),
            np.array([0.0]),
        ]

        result_boxes, result_scores, result_labels = mock_weighted_boxes_fusion(
            boxes_list=boxes,
            scores_list=scores,
            labels_list=labels,
            weights=None,  # Default weights
            iou_thr=0.55,
        )

        assert len(result_boxes) == 1
