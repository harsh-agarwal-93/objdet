"""Celery task definitions for training pipelines.

This module defines the Celery tasks that can be submitted to the
distributed task queue.

Example:
    >>> from objdet.pipelines.tasks import train_model
    >>>
    >>> # Submit training job
    >>> result = train_model.delay(
    ...     config_path="configs/experiment/coco_frcnn.yaml",
    ...     output_dir="/outputs/run_001",
    ... )
    >>> print(f"Task ID: {result.id}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from celery import Task

from objdet.core.logging import get_logger
from objdet.pipelines.celery_app import app

logger = get_logger(__name__)


@app.task(
    bind=True,
    name="objdet.pipelines.tasks.train_model",
    max_retries=3,
    default_retry_delay=60,
)
def train_model(
    self: Task,
    config_path: str,
    output_dir: str,
    checkpoint: str | None = None,
    max_epochs: int | None = None,
    devices: int = 1,
    accelerator: str = "auto",
    extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a detection model.

    This task runs a full training loop using Lightning Trainer.

    Args:
        self: Celery task instance (bound task).
        config_path: Path to LightningCLI config YAML.
        output_dir: Directory for outputs (checkpoints, logs).
        checkpoint: Optional path to resume from checkpoint.
        max_epochs: Override max epochs from config.
        devices: Number of devices to use.
        accelerator: Accelerator type.
        extra_args: Additional CLI arguments.

    Returns:
        Dictionary with training results (best metrics, checkpoint path).
    """
    import yaml
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

    logger.info(f"Starting training task: config={config_path}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if max_epochs is not None:
        config.setdefault("trainer", {})["max_epochs"] = max_epochs

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Setup model and data from config
        from objdet.cli import create_cli

        # Build trainer with callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=output_path / "checkpoints",
                filename="best-{epoch:02d}-{val_mAP:.4f}",
                monitor="val/mAP",
                mode="max",
                save_top_k=3,
            ),
            EarlyStopping(
                monitor="val/mAP",
                patience=10,
                mode="max",
            ),
        ]

        # Setup loggers
        loggers = [
            TensorBoardLogger(save_dir=output_path, name="tensorboard"),
        ]

        # Add MLflow if configured
        import os

        if os.getenv("MLFLOW_TRACKING_URI"):
            loggers.append(
                MLFlowLogger(
                    experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "objdet"),
                    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
                )
            )

        # Create trainer
        trainer = Trainer(
            default_root_dir=str(output_path),
            max_epochs=config.get("trainer", {}).get("max_epochs", 100),
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=loggers,
        )

        # Create CLI and run
        cli = create_cli()

        # Train
        trainer.fit(
            model=cli.model,
            datamodule=cli.datamodule,
            ckpt_path=checkpoint,
        )

        # Get best checkpoint
        best_ckpt = callbacks[0].best_model_path
        best_score = callbacks[0].best_model_score

        result = {
            "status": "completed",
            "best_checkpoint": best_ckpt,
            "best_mAP": float(best_score) if best_score else None,
            "output_dir": str(output_path),
        }

        logger.info(f"Training complete: best_mAP={best_score}")
        return result

    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Retry on failure
        raise self.retry(exc=e)


@app.task(
    bind=True,
    name="objdet.pipelines.tasks.export_model",
)
def export_model(
    self: Task,
    checkpoint_path: str,
    output_path: str,
    export_format: str = "onnx",
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
) -> dict[str, Any]:
    """Export a trained model to optimized format.

    Args:
        self: Celery task instance (bound task).
        checkpoint_path: Path to model checkpoint.
        output_path: Output path for exported model.
        export_format: Target format (onnx, tensorrt, safetensors).
        input_shape: Input tensor shape.

    Returns:
        Dictionary with export results.
    """
    from objdet.optimization import export_model as do_export

    logger.info(f"Exporting model: {checkpoint_path} -> {export_format}")

    try:
        result_path = do_export(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            export_format=export_format,
            input_shape=input_shape,
        )

        return {
            "status": "completed",
            "output_path": str(result_path),
            "format": export_format,
        }

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


@app.task(
    bind=True,
    name="objdet.pipelines.tasks.preprocess_data",
)
def preprocess_data(
    self: Task,
    input_dir: str,
    output_dir: str,
    format_name: str,
    num_workers: int = 4,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Preprocess dataset to LitData format.

    Args:
        self: Celery task instance (bound task).
        input_dir: Source dataset directory.
        output_dir: Output directory.
        format_name: Source format (coco, voc, yolo).
        num_workers: Number of workers.
        class_names: Class names (required for YOLO).

    Returns:
        Dictionary with preprocessing results.
    """
    from objdet.data.preprocessing import convert_to_litdata

    logger.info(f"Preprocessing data: {input_dir} ({format_name})")

    try:
        convert_to_litdata(
            input_dir=input_dir,
            output_dir=output_dir,
            format_name=format_name,
            num_workers=num_workers,
            class_names=class_names,
        )

        return {
            "status": "completed",
            "output_dir": output_dir,
        }

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
