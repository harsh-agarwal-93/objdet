"""SAHI (Slicing Aided Hyper Inference) wrapper.

This module provides slice-based inference for detecting small objects
in large images. Images are split into overlapping tiles, inference is
run on each tile, and results are merged.

Example:
    >>> from objdet.inference import SlicedInference
    >>>
    >>> sahi = SlicedInference(
    ...     model=predictor,
    ...     slice_height=640,
    ...     slice_width=640,
    ...     overlap_ratio=0.2,
    ... )
    >>> results = sahi.predict("large_image.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    from objdet.core.types import DetectionPrediction
    from objdet.inference.predictor import Predictor

logger = get_logger(__name__)


class SlicedInference:
    """Slicing Aided Hyper Inference for large images.

    SAHI splits large images into overlapping tiles, runs detection
    on each tile, and merges the results using NMS or WBF.

    Args:
        predictor: The predictor to use for inference on slices.
        slice_height: Height of each slice in pixels.
        slice_width: Width of each slice in pixels.
        overlap_ratio: Overlap between adjacent slices (0-1).
        merge_method: How to merge overlapping predictions ("nms" or "wbf").
        nms_threshold: IoU threshold for merging.
        include_full_image: Whether to also run on full image.

    Example:
        >>> sahi = SlicedInference(predictor, slice_height=640, slice_width=640)
        >>> results = sahi.predict("aerial_image.jpg")
    """

    def __init__(
        self,
        predictor: Predictor,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio: float = 0.2,
        merge_method: str = "nms",
        nms_threshold: float = 0.5,
        include_full_image: bool = True,
    ) -> None:
        self.predictor = predictor
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.merge_method = merge_method
        self.nms_threshold = nms_threshold
        self.include_full_image = include_full_image

        logger.info(
            f"SlicedInference: slices={slice_width}x{slice_height}, "
            f"overlap={overlap_ratio}, merge={merge_method}"
        )

    def predict(
        self,
        image: str | Path | Tensor,
    ) -> DetectionPrediction:
        """Run sliced inference on a large image.

        Args:
            image: Image path or tensor.

        Returns:
            Merged predictions from all slices.
        """
        # Load image
        if isinstance(image, (str, Path)):
            from PIL import Image as PILImage
            import numpy as np

            image_pil = PILImage.open(image).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = image

        _, img_height, img_width = image_tensor.shape

        # Calculate slice positions
        slices = self._get_slice_positions(img_height, img_width)

        logger.debug(f"Processing {len(slices)} slices for {img_width}x{img_height} image")

        # Collect all predictions
        all_boxes = []
        all_labels = []
        all_scores = []

        # Run inference on each slice
        for x1, y1, x2, y2 in slices:
            slice_tensor = image_tensor[:, y1:y2, x1:x2]

            # Run prediction
            pred = self.predictor.predict(slice_tensor)

            if pred["boxes"].numel() > 0:
                # Offset boxes to original image coordinates
                boxes = pred["boxes"].clone()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1

                all_boxes.append(boxes)
                all_labels.append(pred["labels"])
                all_scores.append(pred["scores"])

        # Optionally include full image prediction
        if self.include_full_image:
            full_pred = self.predictor.predict(image_tensor)
            if full_pred["boxes"].numel() > 0:
                all_boxes.append(full_pred["boxes"])
                all_labels.append(full_pred["labels"])
                all_scores.append(full_pred["scores"])

        # Merge predictions
        if not all_boxes:
            return {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
            }

        merged_boxes = torch.cat(all_boxes, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)
        merged_scores = torch.cat(all_scores, dim=0)

        # Apply merging
        return self._merge_predictions(
            merged_boxes,
            merged_labels,
            merged_scores,
            img_height,
            img_width,
        )

    def _get_slice_positions(
        self,
        img_height: int,
        img_width: int,
    ) -> list[tuple[int, int, int, int]]:
        """Calculate slice positions with overlap.

        Returns:
            List of (x1, y1, x2, y2) slice coordinates.
        """
        slices = []

        step_x = int(self.slice_width * (1 - self.overlap_ratio))
        step_y = int(self.slice_height * (1 - self.overlap_ratio))

        y = 0
        while y < img_height:
            x = 0
            while x < img_width:
                x2 = min(x + self.slice_width, img_width)
                y2 = min(y + self.slice_height, img_height)
                x1 = max(0, x2 - self.slice_width)
                y1 = max(0, y2 - self.slice_height)

                slices.append((x1, y1, x2, y2))
                x += step_x

            y += step_y

        return slices

    def _merge_predictions(
        self,
        boxes: Tensor,
        labels: Tensor,
        scores: Tensor,
        img_height: int,
        img_width: int,
    ) -> DetectionPrediction:
        """Merge overlapping predictions from multiple slices."""
        if self.merge_method == "nms":
            return self._merge_nms(boxes, labels, scores)
        elif self.merge_method == "wbf":
            return self._merge_wbf(boxes, labels, scores, img_height, img_width)
        else:
            raise ValueError(f"Unknown merge method: {self.merge_method}")

    def _merge_nms(
        self,
        boxes: Tensor,
        labels: Tensor,
        scores: Tensor,
    ) -> DetectionPrediction:
        """Merge using class-wise NMS."""
        from torchvision.ops import batched_nms

        keep = batched_nms(boxes, scores, labels, self.nms_threshold)

        return {
            "boxes": boxes[keep],
            "labels": labels[keep],
            "scores": scores[keep],
        }

    def _merge_wbf(
        self,
        boxes: Tensor,
        labels: Tensor,
        scores: Tensor,
        img_height: int,
        img_width: int,
    ) -> DetectionPrediction:
        """Merge using Weighted Box Fusion."""
        try:
            from ensemble_boxes import weighted_boxes_fusion
        except ImportError:
            logger.warning("ensemble-boxes not installed, falling back to NMS")
            return self._merge_nms(boxes, labels, scores)

        # Normalize boxes
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] /= img_width
        boxes_norm[:, [1, 3]] /= img_height
        boxes_norm = boxes_norm.clamp(0, 1)

        # WBF expects list format
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [boxes_norm.numpy().tolist()],
            [scores.numpy().tolist()],
            [labels.numpy().tolist()],
            iou_thr=self.nms_threshold,
        )

        # Denormalize
        result_boxes = torch.tensor(fused_boxes, dtype=torch.float32)
        if result_boxes.numel() > 0:
            result_boxes[:, [0, 2]] *= img_width
            result_boxes[:, [1, 3]] *= img_height

        return {
            "boxes": result_boxes,
            "labels": torch.tensor(fused_labels, dtype=torch.int64),
            "scores": torch.tensor(fused_scores, dtype=torch.float32),
        }
