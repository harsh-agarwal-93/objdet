"""Pydantic models for training-related API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training job configuration."""

    name: str = Field(..., description="Training run name")
    model_architecture: str = Field(..., description="Model architecture (yolov8, resnet, etc.)")
    dataset: str = Field(..., description="Dataset name")
    epochs: int = Field(100, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Batch size")
    learning_rate: float = Field(0.001, gt=0, description="Learning rate")
    optimizer: str = Field("adam", description="Optimizer (adam, sgd, adamw)")
    gpu: str = Field("auto", description="GPU selection (auto, gpu0, gpu1, multi)")
    priority: str = Field("normal", description="Job priority (low, normal, high)")
    mixed_precision: str = Field("fp16", description="Mixed precision (fp16, fp32, bf16)")
    save_checkpoints: bool = Field(True, description="Save model checkpoints")
    early_stopping: bool = Field(True, description="Enable early stopping")
    log_to_mlflow: bool = Field(True, description="Log to MLFlow")
    data_augmentation: bool = Field(True, description="Enable data augmentation")
    config_path: str | None = Field(None, description="Optional config file path")
    output_dir: str | None = Field(None, description="Custom output directory")


class TrainingJobResponse(BaseModel):
    """Response after submitting a training job."""

    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Initial job status")
    created_at: str = Field(..., description="Job creation timestamp")
    estimated_duration: str | None = Field(None, description="Estimated completion time")


class TaskStatusResponse(BaseModel):
    """Celery task status response."""

    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] | None = None  # current_epoch, total_epochs, etc.


class TrainingRun(BaseModel):
    """Training run information from MLFlow."""

    run_id: str
    run_name: str
    experiment_id: str
    status: str
    start_time: str | None = None
    end_time: str | None = None
    metrics: dict[str, float] | None = None
    params: dict[str, Any] | None = None
    tags: dict[str, str] | None = None
