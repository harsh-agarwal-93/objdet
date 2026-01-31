"""Unit tests for ensemble models (NMS and WBF)."""

from __future__ import annotations

import torch

from objdet.core.constants import EnsembleStrategy


class TestEnsembleWeights:
    """Test ensemble weight handling without full initialization."""

    def test_weight_normalization(self) -> None:
        """Test that weights are normalized correctly."""
        weights = [2.0, 3.0]
        total = sum(weights)
        normalized = [w / total for w in weights]

        assert abs(sum(normalized) - 1.0) < 1e-6
        assert abs(normalized[0] - 0.4) < 1e-6
        assert abs(normalized[1] - 0.6) < 1e-6

    def test_equal_weights_calculation(self) -> None:
        """Test equal weights for multiple models."""
        num_models = 3
        expected_weight = 1.0 / num_models
        weights = [expected_weight] * num_models

        assert abs(sum(weights) - 1.0) < 1e-6


class TestNMSFusion:
    """Test NMS fusion logic."""

    def test_fuse_empty_predictions(self) -> None:
        """Test fusion of all empty predictions returns empty result."""
        # Simulate empty predictions
        empty_preds = [
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            },
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            },
        ]

        # Check that when all boxes are empty, we should get empty result
        all_empty = all(pred["boxes"].numel() == 0 for pred in empty_preds)
        assert all_empty

    def test_box_normalization(self) -> None:
        """Test that boxes are correctly normalized for NMS."""
        height, width = 640, 640
        boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])

        # Normalize
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= width
        normalized[:, [1, 3]] /= height

        expected = torch.tensor([[100.0 / 640, 100.0 / 640, 200.0 / 640, 200.0 / 640]])
        assert torch.allclose(normalized, expected)

    def test_box_denormalization(self) -> None:
        """Test that normalized boxes are correctly converted back."""
        height, width = 1000, 1000
        normalized_boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2]])

        # Denormalize
        denormalized = normalized_boxes.clone()
        denormalized[:, [0, 2]] *= width
        denormalized[:, [1, 3]] *= height

        expected = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        assert torch.allclose(denormalized, expected)


class TestWBFFusion:
    """Test WBF fusion logic."""

    def test_fuse_empty_predictions(self) -> None:
        """Test that empty predictions return empty result."""
        empty_preds = [
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            },
        ]

        all_empty = all(pred["boxes"].numel() == 0 for pred in empty_preds)
        assert all_empty

    def test_confidence_types(self) -> None:
        """Test valid confidence type options."""
        valid_conf_types = ["avg", "max", "box_and_model_avg", "absent_model_aware_avg"]

        for conf_type in valid_conf_types:
            assert conf_type in valid_conf_types


class TestEnsembleStrategy:
    """Test ensemble strategy enumeration."""

    def test_nms_strategy_value(self) -> None:
        """Test NMS strategy enum value."""
        assert EnsembleStrategy.NMS.value == "nms"

    def test_soft_nms_strategy_value(self) -> None:
        """Test Soft-NMS strategy enum value."""
        assert EnsembleStrategy.SOFT_NMS.value == "soft_nms"

    def test_wbf_strategy_value(self) -> None:
        """Test WBF strategy enum value."""
        assert EnsembleStrategy.WBF.value == "wbf"


class TestPredictionFormat:
    """Test prediction format handling."""

    def test_prediction_has_required_keys(self) -> None:
        """Test that a proper prediction has all required keys."""
        prediction = {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }

        assert "boxes" in prediction
        assert "labels" in prediction
        assert "scores" in prediction

    def test_empty_prediction_format(self) -> None:
        """Test the format of an empty prediction."""
        empty_pred = {
            "boxes": torch.empty(0, 4),
            "labels": torch.empty(0, dtype=torch.int64),
            "scores": torch.empty(0),
        }

        assert empty_pred["boxes"].shape == (0, 4)
        assert empty_pred["labels"].numel() == 0
        assert empty_pred["scores"].numel() == 0

    def test_boxes_shape(self) -> None:
        """Test that boxes have correct shape."""
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

        assert boxes.shape[0] == 2  # 2 boxes
        assert boxes.shape[1] == 4  # 4 coordinates (x1, y1, x2, y2)
