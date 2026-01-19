"""Model export utilities.

This module provides functions for exporting detection models to
various optimized formats:
- ONNX for portable inference
- SafeTensors for efficient weight storage
- TensorRT for GPU-optimized inference

Example:
    >>> from objdet.optimization import export_to_onnx
    >>>
    >>> export_to_onnx(
    ...     checkpoint_path="model.ckpt",
    ...     output_path="model.onnx",
    ...     input_shape=(1, 3, 640, 640),
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from objdet.core.exceptions import ExportError
from objdet.core.logging import get_logger

if TYPE_CHECKING:
    from objdet.models.base import BaseLightningDetector

logger = get_logger(__name__)


def export_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    export_format: Literal["onnx", "tensorrt", "safetensors"] = "onnx",
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Export model to specified format.

    This is the main entry point for the CLI export command.

    Args:
        checkpoint_path: Path to model checkpoint.
        output_path: Output path for exported model.
        export_format: Target format ("onnx", "tensorrt", "safetensors").
        config_path: Optional model config file.
        **kwargs: Additional format-specific arguments.

    Returns:
        Path to the exported model file.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise ExportError(
            f"Checkpoint not found: {checkpoint_path}",
            source_format="pytorch",
            target_format=export_format,
        )

    # Load model
    model = _load_model_from_checkpoint(checkpoint_path, config_path)

    # Export based on format
    if export_format.lower() == "onnx":
        return export_to_onnx(model, output_path, **kwargs)
    elif export_format.lower() == "safetensors":
        return export_to_safetensors(model, output_path, **kwargs)
    elif export_format.lower() == "tensorrt":
        return export_to_tensorrt(model, output_path, **kwargs)
    else:
        raise ExportError(
            f"Unknown export format: {export_format}",
            source_format="pytorch",
            target_format=export_format,
        )


def export_to_onnx(
    model: BaseLightningDetector | Path,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    simplify: bool = True,
) -> Path:
    """Export model to ONNX format.

    Args:
        model: Model instance or checkpoint path.
        output_path: Output path for ONNX model.
        input_shape: Input tensor shape (batch, channels, height, width).
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axis configuration for variable batch/size.
        simplify: Whether to simplify the ONNX graph.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model if path
    if isinstance(model, (str, Path)):
        model = _load_model_from_checkpoint(Path(model))

    model.eval()

    # Create sample input
    sample_input = torch.randn(*input_shape)

    # Default dynamic axes for detection
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "boxes": {0: "batch_size", 1: "num_detections"},
            "labels": {0: "batch_size", 1: "num_detections"},
            "scores": {0: "batch_size", 1: "num_detections"},
        }

    logger.info(f"Exporting to ONNX: {output_path}")

    # Export
    torch.onnx.export(
        model,
        [sample_input],
        str(output_path),
        input_names=["input"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(output_path))
            simplified, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified, str(output_path))
                logger.info("ONNX model simplified successfully")
        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")

    logger.info(f"ONNX export complete: {output_path}")
    return output_path


def export_to_safetensors(
    model: BaseLightningDetector | Path,
    output_path: str | Path,
    include_metadata: bool = True,
) -> Path:
    """Export model weights to SafeTensors format.

    SafeTensors is a fast and safe format for storing tensors.

    Args:
        model: Model instance or checkpoint path.
        output_path: Output path for SafeTensors file.
        include_metadata: Whether to include model metadata.

    Returns:
        Path to the exported SafeTensors file.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise ExportError(
            "safetensors is required. Install with: pip install safetensors",
            source_format="pytorch",
            target_format="safetensors",
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model if path
    if isinstance(model, (str, Path)):
        model = _load_model_from_checkpoint(Path(model))

    # Get state dict
    state_dict = model.state_dict()

    # Build metadata
    metadata = {}
    if include_metadata:
        metadata = {
            "model_type": model.__class__.__name__,
            "num_classes": str(model.num_classes),
            "class_index_mode": str(model.class_index_mode.value),
        }

    logger.info(f"Exporting to SafeTensors: {output_path}")
    save_file(state_dict, str(output_path), metadata=metadata)

    logger.info(f"SafeTensors export complete: {output_path}")
    return output_path


def export_to_tensorrt(
    model: BaseLightningDetector | Path,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    fp16: bool = True,
    int8: bool = False,
) -> Path:
    """Export model to TensorRT format.

    Note: Requires torch-tensorrt to be installed.

    Args:
        model: Model instance or checkpoint path.
        output_path: Output path for TensorRT engine.
        input_shape: Input tensor shape.
        fp16: Enable FP16 precision.
        int8: Enable INT8 precision (requires calibration).

    Returns:
        Path to the exported TensorRT engine.
    """
    try:
        import torch_tensorrt
    except ImportError as e:
        raise ExportError(
            "torch-tensorrt is required for TensorRT export",
            source_format="pytorch",
            target_format="tensorrt",
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model if path
    if isinstance(model, (str, Path)):
        model = _load_model_from_checkpoint(Path(model))

    model.eval().cuda()

    # Compile with TensorRT
    logger.info(f"Compiling with TensorRT: {output_path}")

    enabled_precisions = {torch.float32}
    if fp16:
        enabled_precisions.add(torch.float16)
    if int8:
        enabled_precisions.add(torch.int8)

    inputs = [
        torch_tensorrt.Input(
            shape=input_shape,
            dtype=torch.float32,
        )
    ]

    trt_model = torch_tensorrt.compile(
        model,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
    )

    # Save
    torch.jit.save(trt_model, str(output_path))

    logger.info(f"TensorRT export complete: {output_path}")
    return output_path


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> BaseLightningDetector:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Try to determine model class
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]

        # Get model class from registry
        from objdet.models import MODEL_REGISTRY

        model_type = hparams.get("model_type", "").lower()
        if model_type in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY.get(model_type)
            return model_class.load_from_checkpoint(checkpoint_path)

    raise ExportError(
        "Cannot determine model class from checkpoint",
        source_format="pytorch",
        target_format="unknown",
    )
