"""Prediction-only dataset for inference without labels.

This module provides a simple dataset for running inference on
a directory of images without requiring annotation files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class PredictDataset(Dataset):
    """Dataset for prediction/inference without labels.

    Args:
        image_dir: Directory containing images.
        transforms: Optional transform to apply.
        extensions: Image file extensions to include.
    """

    def __init__(
        self,
        image_dir: str | Path,
        transforms: Callable | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        # Find all images
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.image_dir.glob(f"*{ext.upper()}"))

        self.image_paths = sorted(set(self.image_paths))

        logger.info(f"Loaded predict dataset: {len(self.image_paths)} images")

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        """Get image and metadata by index.

        Returns:
            Tuple of (image_tensor, metadata_dict).
            Metadata contains image_id and original file path.
        """
        img_path = self.image_paths[index]

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # Convert to tensor
        image_tensor = (
            torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        )

        # Metadata (not a standard target, just info)
        metadata = {
            "image_id": index,
            "file_path": str(img_path),
            "original_size": original_size,
        }

        # Apply transforms (only to image for predict)
        if self.transforms is not None:
            # For predict, we pass an empty target
            empty_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }
            image_tensor, _ = self.transforms(image_tensor, empty_target)

        return image_tensor, metadata
