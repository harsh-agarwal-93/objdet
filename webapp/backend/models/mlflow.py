"""Pydantic models for MLFlow data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Experiment(BaseModel):
    """MLFlow experiment."""

    experiment_id: str
    name: str
    artifact_location: str | None = None
    lifecycle_stage: str | None = None
    tags: dict[str, str] | None = None


class Run(BaseModel):
    """MLFlow run."""

    run_id: str
    run_name: str | None = None
    experiment_id: str
    status: str
    start_time: int | None = None
    end_time: int | None = None
    artifact_uri: str | None = None


class RunMetrics(BaseModel):
    """Run metrics time series."""

    run_id: str
    metrics: list[dict[str, Any]] = Field(default_factory=list)


class Artifact(BaseModel):
    """MLFlow artifact metadata."""

    path: str
    is_dir: bool
    file_size: int | None = None


class ModelVersion(BaseModel):
    """Registered model version."""

    name: str
    version: str
    creation_timestamp: int
    last_updated_timestamp: int
    current_stage: str
    description: str | None = None
    source: str | None = None
    run_id: str | None = None
