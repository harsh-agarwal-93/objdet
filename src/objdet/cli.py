"""LightningCLI configuration for ObjDet.

This module provides the command-line interface for ObjDet using
PyTorch Lightning's CLI system. It supports subcommands for training,
validation, testing, prediction, and serving.

Example:
    >>> # Train a model
    >>> objdet fit --config configs/experiment/faster_rcnn_coco.yaml
    >>>
    >>> # Validate a trained model
    >>> objdet validate --config configs/experiment/faster_rcnn_coco.yaml --ckpt_path best.ckpt
    >>>
    >>> # Run prediction
    >>> objdet predict --config configs/experiment/faster_rcnn_coco.yaml --ckpt_path best.ckpt
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from objdet.core.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from lightning import LightningDataModule, LightningModule

logger = get_logger(__name__)


class ObjDetCLI(LightningCLI):
    """Custom LightningCLI for ObjDet.

    Extends the base LightningCLI with ObjDet-specific features:
    - Custom subcommands for serving and export
    - Automatic logging configuration
    - Support for ObjDet config structure

    Example:
        >>> cli = ObjDetCLI(
        ...     model_class=BaseLightningDetector,
        ...     datamodule_class=BaseDataModule,
        ...     parser_kwargs={"parser_mode": "omegaconf"},
        ... )
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the CLI parser.

        Args:
            parser: The Lightning argument parser.
        """
        # Logging configuration
        parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            help="Logging level (DEBUG, INFO, WARNING, ERROR)",
        )
        parser.add_argument(
            "--log_format",
            type=str,
            default="rich",
            choices=["rich", "json"],
            help="Log output format",
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default=None,
            help="Directory for log files (optional)",
        )

        # Link common arguments
        parser.link_arguments("log_level", "trainer.logger.init_args.log_level", apply_on="parse")

    def before_instantiate_classes(self) -> None:
        """Configure logging before instantiating model and data classes."""
        config = self.config[self.config.subcommand]

        # Configure logging
        configure_logging(
            level=getattr(config, "log_level", "INFO"),
            log_format=getattr(config, "log_format", "rich"),
            log_dir=getattr(config, "log_dir", None),
        )

        logger.info(
            f"ObjDet CLI starting: subcommand={self.config.subcommand}",
        )

    def after_fit(self) -> None:
        """Called after fit completes."""
        logger.info("Training completed successfully")

    def after_validate(self) -> None:
        """Called after validation completes."""
        logger.info("Validation completed")

    def after_test(self) -> None:
        """Called after testing completes."""
        logger.info("Testing completed")

    def after_predict(self) -> None:
        """Called after prediction completes."""
        logger.info("Prediction completed")


def create_cli(
    model_class: type[LightningModule] | None = None,
    datamodule_class: type[LightningDataModule] | None = None,
    **kwargs: Any,
) -> ObjDetCLI:
    """Create and return an ObjDetCLI instance.

    This factory function creates a CLI with sensible defaults for ObjDet.

    Args:
        model_class: Optional model class to use. If None, will be specified via config.
        datamodule_class: Optional datamodule class. If None, will be specified via config.
        **kwargs: Additional arguments passed to ObjDetCLI.

    Returns:
        Configured ObjDetCLI instance.

    Example:
        >>> from objdet.models import FasterRCNN
        >>> from objdet.data import COCODataModule
        >>>
        >>> cli = create_cli(
        ...     model_class=FasterRCNN,
        ...     datamodule_class=COCODataModule,
        ... )
    """
    default_kwargs = {
        "save_config_kwargs": {"overwrite": True},
        "parser_kwargs": {
            "parser_mode": "omegaconf",
            "default_config_files": [],
        },
    }
    default_kwargs.update(kwargs)

    return ObjDetCLI(
        model_class=model_class,
        datamodule_class=datamodule_class,
        subclass_mode_model=model_class is None,
        subclass_mode_data=datamodule_class is None,
        **default_kwargs,
    )


def main() -> None:
    """Main entry point for the ObjDet CLI.

    This function is called when running:
    - `objdet <command>` (via console script)
    - `python -m objdet <command>` (via __main__.py)
    """
    # Check for custom subcommands before delegating to LightningCLI
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]

        if subcommand == "serve":
            _run_serve_command()
            return

        if subcommand == "export":
            _run_export_command()
            return

        if subcommand == "preprocess":
            _run_preprocess_command()
            return

    # Standard Lightning CLI commands (fit, validate, test, predict)
    # Import base classes here to avoid circular imports at module load
    try:
        create_cli()
    except Exception as e:
        logger.exception(f"CLI error: {e}")
        sys.exit(1)


def _run_serve_command() -> None:
    """Run the serving subcommand."""
    # Parse serve-specific arguments
    import argparse

    from objdet.serving.server import run_server

    parser = argparse.ArgumentParser(description="Start ObjDet inference server")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to serving configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers per device",
    )

    args = parser.parse_args(sys.argv[2:])
    run_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
        workers_per_device=args.workers,
    )


def _run_export_command() -> None:
    """Run the export subcommand."""
    import argparse

    parser = argparse.ArgumentParser(description="Export model to optimized format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported model",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tensorrt", "safetensors"],
        default="onnx",
        help="Export format",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Model configuration file (optional)",
    )

    args = parser.parse_args(sys.argv[2:])

    # Import and run export
    from objdet.optimization import export_model

    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        export_format=args.format,
        config_path=args.config,
    )


def _run_preprocess_command() -> None:
    """Run the preprocess subcommand for LitData conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset to LitData format")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input dataset directory (optional if format-specific paths provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for LitData dataset",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "voc", "yolo"],
        required=True,
        help="Input dataset format",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of preprocessing workers",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="List of class names (required for YOLO format)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Dataset splits to convert (default: train val)",
    )
    # COCO-specific paths
    parser.add_argument(
        "--coco_ann_file",
        type=str,
        default=None,
        help="COCO annotation file path",
    )
    parser.add_argument(
        "--coco_images_dir",
        type=str,
        default=None,
        help="COCO images directory",
    )
    # VOC-specific paths
    parser.add_argument(
        "--voc_images_dir",
        type=str,
        default=None,
        help="VOC JPEGImages directory",
    )
    parser.add_argument(
        "--voc_annotations_dir",
        type=str,
        default=None,
        help="VOC Annotations directory",
    )
    parser.add_argument(
        "--voc_imagesets_dir",
        type=str,
        default=None,
        help="VOC ImageSets/Main directory",
    )
    # YOLO-specific paths
    parser.add_argument(
        "--yolo_images_dir",
        type=str,
        default=None,
        help="YOLO images directory",
    )
    parser.add_argument(
        "--yolo_labels_dir",
        type=str,
        default=None,
        help="YOLO labels directory",
    )

    args = parser.parse_args(sys.argv[2:])

    # Import and run preprocessing
    from objdet.data.preprocessing import convert_to_litdata

    convert_to_litdata(
        output_dir=args.output,
        format_name=args.format,
        input_dir=args.input,
        num_workers=args.num_workers,
        class_names=args.class_names,
        splits=args.splits,
        coco_ann_file=args.coco_ann_file,
        coco_images_dir=args.coco_images_dir,
        voc_images_dir=args.voc_images_dir,
        voc_annotations_dir=args.voc_annotations_dir,
        voc_imagesets_dir=args.voc_imagesets_dir,
        yolo_images_dir=args.yolo_images_dir,
        yolo_labels_dir=args.yolo_labels_dir,
    )


if __name__ == "__main__":
    main()
