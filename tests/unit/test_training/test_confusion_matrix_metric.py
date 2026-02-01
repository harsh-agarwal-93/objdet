"""Unit tests for ConfusionMatrix metric."""

from __future__ import annotations

import pytest
import torch


class TestConfusionMatrix:
    """Tests for ConfusionMatrix metric class."""

    @pytest.fixture
    def confusion_matrix(self):
        """Create a ConfusionMatrix instance."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        return ConfusionMatrix(num_classes=3, iou_threshold=0.5, confidence_threshold=0.25)

    def test_init(self) -> None:
        """Test initialization with parameters."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        cm = ConfusionMatrix(
            num_classes=5,
            iou_threshold=0.6,
            confidence_threshold=0.3,
        )

        assert cm.num_classes == 5
        assert cm.iou_threshold == 0.6
        assert cm.confidence_threshold == 0.3
        # Matrix should be (num_classes+1) x (num_classes+1) for background
        assert cm.matrix.shape == (6, 6)

    def test_init_default_thresholds(self) -> None:
        """Test initialization with default thresholds."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        cm = ConfusionMatrix(num_classes=10)

        assert cm.iou_threshold == 0.5
        assert cm.confidence_threshold == 0.25

    def test_update_single_image(self, confusion_matrix) -> None:
        """Test updating with a single image prediction."""
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        confusion_matrix.update(preds, targets)

        # Should have one true positive for class 0
        assert confusion_matrix.matrix[0, 0] == 1

    def test_update_batch(self, confusion_matrix) -> None:
        """Test updating with a batch of predictions."""
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.8]),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                "labels": torch.tensor([1]),
            },
        ]

        confusion_matrix.update(preds, targets)

        # Should have two true positives
        assert confusion_matrix.matrix[0, 0] == 1
        assert confusion_matrix.matrix[1, 1] == 1

    def test_box_iou_computation(self, confusion_matrix) -> None:
        """Test IoU computation between two boxes."""
        box1 = torch.tensor([0.0, 0.0, 100.0, 100.0])
        box2 = torch.tensor([50.0, 50.0, 150.0, 150.0])

        iou = confusion_matrix._box_iou(box1, box2)

        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 1e-4

    def test_box_iou_no_overlap(self, confusion_matrix) -> None:
        """Test IoU for non-overlapping boxes."""
        box1 = torch.tensor([0.0, 0.0, 50.0, 50.0])
        box2 = torch.tensor([100.0, 100.0, 150.0, 150.0])

        iou = confusion_matrix._box_iou(box1, box2)

        assert iou == 0.0

    def test_box_iou_perfect_overlap(self, confusion_matrix) -> None:
        """Test IoU for identical boxes."""
        box = torch.tensor([10.0, 10.0, 50.0, 50.0])

        iou = confusion_matrix._box_iou(box, box)

        assert abs(iou - 1.0) < 1e-6

    def test_compute_returns_matrix(self, confusion_matrix) -> None:
        """Test that compute returns the confusion matrix."""
        result = confusion_matrix.compute()

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 4)  # 3 classes + 1 background

    def test_false_positive_detection(self) -> None:
        """Test that false positives are counted correctly."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        cm = ConfusionMatrix(num_classes=2)

        # Prediction with no matching ground truth
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            }
        ]
        targets = [
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
            }
        ]

        cm.update(preds, targets)

        # False positive: prediction at background row (num_classes), predicted class column
        assert cm.matrix[cm.num_classes, 0] == 1

    def test_false_negative_detection(self) -> None:
        """Test that false negatives are counted correctly."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        cm = ConfusionMatrix(num_classes=2)

        # Ground truth with no matching prediction
        preds = [
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        cm.update(preds, targets)

        # False negative: ground truth class row, background column (num_classes)
        assert cm.matrix[1, cm.num_classes] == 1

    def test_confidence_threshold_filtering(self) -> None:
        """Test that low-confidence predictions are filtered."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        cm = ConfusionMatrix(num_classes=2, confidence_threshold=0.5)

        # Low confidence prediction
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.3]),  # Below threshold
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        cm.update(preds, targets)

        # No true positive because score is below threshold
        assert cm.matrix[0, 0] == 0
        # Should be a false negative
        assert cm.matrix[0, cm.num_classes] == 1

    def test_iou_threshold_matching(self) -> None:
        """Test that IoU threshold affects matching."""
        from objdet.training.metrics.confusion_matrix import ConfusionMatrix

        cm = ConfusionMatrix(num_classes=2, iou_threshold=0.9)

        # Prediction with some overlap but below 0.9 IoU
        preds = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        cm.update(preds, targets)

        # With high IoU threshold, these shouldn't match
        # Result: 1 FP + 1 FN instead of 1 TP
        assert cm.matrix[0, 0] == 0  # No TP
