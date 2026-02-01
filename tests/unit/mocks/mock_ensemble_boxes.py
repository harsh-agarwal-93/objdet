"""Mock ensemble_boxes module for unit testing.

This module provides mock implementations of ensemble_boxes functions
(NMS, Soft-NMS, WBF) to enable unit testing without the dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mock_nms(
    boxes_list: list[NDArray[np.floating]],
    scores_list: list[NDArray[np.floating]],
    labels_list: list[NDArray[np.floating]],
    iou_thr: float = 0.5,
    weights: list[float] | None = None,
    skip_box_thr: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Mock NMS ensemble that merges boxes from multiple models.

    This simplified mock:
    1. Concatenates all boxes from all models
    2. Sorts by score (descending)
    3. Returns top boxes without actual NMS suppression

    Args:
        boxes_list: List of box arrays from each model, shape (N, 4) in [0, 1].
        scores_list: List of score arrays from each model, shape (N,).
        labels_list: List of label arrays from each model, shape (N,).
        iou_thr: IoU threshold for suppression.
        weights: Model weights (ignored in mock).
        skip_box_thr: Skip boxes below this score.

    Returns:
        Tuple of (boxes, scores, labels) after mock NMS.
    """
    if not boxes_list or all(len(b) == 0 for b in boxes_list):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Concatenate all predictions
    all_boxes = np.concatenate([b for b in boxes_list if len(b) > 0], axis=0)
    all_scores = np.concatenate([s for s in scores_list if len(s) > 0], axis=0)
    all_labels = np.concatenate([lbl for lbl in labels_list if len(lbl) > 0], axis=0)

    # Filter by threshold
    mask = all_scores >= skip_box_thr
    all_boxes = all_boxes[mask]
    all_scores = all_scores[mask]
    all_labels = all_labels[mask]

    if len(all_boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Sort by score and take unique class predictions
    order = np.argsort(-all_scores)
    all_boxes = all_boxes[order]
    all_scores = all_scores[order]
    all_labels = all_labels[order]

    # Simple deduplication: keep first occurrence per class
    seen_classes: set[int] = set()
    keep_indices = []
    for i, label in enumerate(all_labels):
        label_int = int(label)
        if label_int not in seen_classes:
            seen_classes.add(label_int)
            keep_indices.append(i)

    if keep_indices:
        return (
            all_boxes[keep_indices],
            all_scores[keep_indices],
            all_labels[keep_indices],
        )

    return (
        all_boxes[:1],
        all_scores[:1],
        all_labels[:1],
    )


def mock_soft_nms(
    boxes_list: list[NDArray[np.floating]],
    scores_list: list[NDArray[np.floating]],
    labels_list: list[NDArray[np.floating]],
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    thresh: float = 0.001,
    weights: list[float] | None = None,
    skip_box_thr: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Mock Soft-NMS ensemble.

    Uses the same logic as mock_nms for simplicity.

    Args:
        boxes_list: List of box arrays from each model.
        scores_list: List of score arrays from each model.
        labels_list: List of label arrays from each model.
        iou_thr: IoU threshold.
        sigma: Gaussian sigma for score decay (ignored in mock).
        thresh: Score threshold for removal (ignored in mock).
        weights: Model weights (ignored in mock).
        skip_box_thr: Skip boxes below this score.

    Returns:
        Tuple of (boxes, scores, labels) after mock Soft-NMS.
    """
    return mock_nms(
        boxes_list=boxes_list,
        scores_list=scores_list,
        labels_list=labels_list,
        iou_thr=iou_thr,
        weights=weights,
        skip_box_thr=skip_box_thr,
    )


def mock_weighted_boxes_fusion(
    boxes_list: list[NDArray[np.floating]],
    scores_list: list[NDArray[np.floating]],
    labels_list: list[NDArray[np.floating]],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    weights: list[float] | None = None,
    conf_type: str = "avg",
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Mock Weighted Box Fusion ensemble.

    This simplified mock:
    1. Concatenates boxes from all models
    2. Groups overlapping boxes by class
    3. Returns weighted average of grouped boxes

    Args:
        boxes_list: List of box arrays from each model, shape (N, 4) in [0, 1].
        scores_list: List of score arrays from each model, shape (N,).
        labels_list: List of label arrays from each model, shape (N,).
        iou_thr: IoU threshold for grouping.
        skip_box_thr: Skip boxes below this score.
        weights: Model weights for fusion.
        conf_type: Confidence aggregation type (ignored in mock).

    Returns:
        Tuple of (boxes, scores, labels) after mock WBF.
    """
    if not boxes_list or all(len(b) == 0 for b in boxes_list):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Apply weights if provided
    if weights is None:
        weights = [1.0] * len(boxes_list)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Concatenate all predictions with weights applied to scores
    weighted_boxes = []
    weighted_scores = []
    all_labels_list = []

    for boxes, scores, labels, weight in zip(
        boxes_list, scores_list, labels_list, weights, strict=True
    ):
        if len(boxes) > 0:
            weighted_boxes.append(boxes)
            weighted_scores.append(scores * weight)
            all_labels_list.append(labels)

    if not weighted_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    all_boxes = np.concatenate(weighted_boxes, axis=0)
    all_scores = np.concatenate(weighted_scores, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    # Filter by threshold
    mask = all_scores >= skip_box_thr
    all_boxes = all_boxes[mask]
    all_scores = all_scores[mask]
    all_labels = all_labels[mask]

    if len(all_boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Group by class and average
    unique_labels = np.unique(all_labels)
    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for label in unique_labels:
        class_mask = all_labels == label
        class_boxes = all_boxes[class_mask]
        class_scores = all_scores[class_mask]

        # Average boxes weighted by scores
        if len(class_boxes) > 0:
            score_weights = class_scores / class_scores.sum()
            avg_box = np.sum(class_boxes * score_weights[:, None], axis=0)
            avg_score = np.mean(class_scores)

            fused_boxes.append(avg_box)
            fused_scores.append(avg_score)
            fused_labels.append(label)

    if fused_boxes:
        return (
            np.array(fused_boxes, dtype=np.float32),
            np.array(fused_scores, dtype=np.float32),
            np.array(fused_labels, dtype=np.float32),
        )

    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )
