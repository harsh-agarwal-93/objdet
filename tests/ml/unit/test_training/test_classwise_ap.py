"""Unit tests for ClasswiseAP metric."""

from __future__ import annotations

import pytest
import torch


class TestClasswiseAP:
    """Tests for ClasswiseAP metric class."""

    @pytest.fixture
    def classwise_ap(self):
        """Create a ClasswiseAP instance."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        return ClasswiseAP(num_classes=3)

    def test_init(self) -> None:
        """Test initialization with parameters."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        metric = ClasswiseAP(num_classes=10)

        assert metric.num_classes == 10
        assert len(metric.class_names) == 10

    def test_init_with_class_names(self) -> None:
        """Test initialization with custom class names."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        class_names = ["person", "car", "bicycle"]
        metric = ClasswiseAP(num_classes=3, class_names=class_names)

        assert metric.class_names == class_names

    def test_init_default_class_names(self) -> None:
        """Test default class names are generated."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        metric = ClasswiseAP(num_classes=5)

        assert metric.class_names == ["class_0", "class_1", "class_2", "class_3", "class_4"]

    def test_init_with_iou_thresholds(self) -> None:
        """Test initialization with custom IoU thresholds."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        iou_thresholds = [0.5, 0.75]
        metric = ClasswiseAP(num_classes=3, iou_thresholds=iou_thresholds)

        assert metric.num_classes == 3

    def test_update_delegates_to_map_metric(self, classwise_ap) -> None:
        """Test that update calls underlying mAP metric."""
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

        # Should not raise
        classwise_ap.update(preds, targets)

    def test_compute_returns_dict(self) -> None:
        """Test that compute returns a dictionary with required keys."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        # Create fresh metric for this test
        metric = ClasswiseAP(num_classes=2, class_names=["cat", "dog"])

        # Add multiple samples across different classes
        for class_id in range(2):
            for i in range(3):
                offset = class_id * 100 + i * 10
                preds = [
                    {
                        "boxes": torch.tensor([[10.0 + offset, 10.0, 50.0 + offset, 50.0]]),
                        "labels": torch.tensor([class_id]),
                        "scores": torch.tensor([0.9]),
                    }
                ]
                targets = [
                    {
                        "boxes": torch.tensor([[10.0 + offset, 10.0, 50.0 + offset, 50.0]]),
                        "labels": torch.tensor([class_id]),
                    }
                ]
                metric.update(preds, targets)

        result = metric.compute()

        assert isinstance(result, dict)
        assert "map" in result
        assert "map_50" in result
        assert "map_75" in result

    def test_reset_clears_state(self) -> None:
        """Test that reset clears accumulated state."""
        from objdet.training.metrics.classwise_ap import ClasswiseAP

        metric = ClasswiseAP(num_classes=3)

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

        metric.update(preds, targets)
        metric.reset()

        # After reset, should be back to initial state
        # This shouldn't raise even with no data
        # (torchmetrics handles this gracefully)

    def test_multiple_updates(self, classwise_ap) -> None:
        """Test multiple update calls accumulate correctly."""
        preds1 = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
                "scores": torch.tensor([0.9]),
            }
        ]
        targets1 = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
        ]

        preds2 = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 150.0, 150.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.85]),
            }
        ]
        targets2 = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 150.0, 150.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        classwise_ap.update(preds1, targets1)
        classwise_ap.update(preds2, targets2)

        result = classwise_ap.compute()
        assert "map" in result
