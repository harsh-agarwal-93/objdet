"""MLFlow service for data retrieval."""

from __future__ import annotations

from typing import Any

import polars as pl
from mlflow.client import MlflowClient

from backend.core.config import settings


def get_mlflow_client() -> MlflowClient:
    """Get configured MLFlow client.

    Returns:
        MLFlow client instance.
    """
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def list_experiments() -> list[dict[str, Any]]:
    """List all MLFlow experiments.

    Returns:
        List of experiment dictionaries.
    """
    client = get_mlflow_client()
    experiments = client.search_experiments()

    return [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "tags": exp.tags,
        }
        for exp in experiments
    ]


def list_runs(
    experiment_id: str | None = None,
    status: str | None = None,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """List MLFlow runs with optional filtering.

    Args:
        experiment_id: Filter by experiment ID.
        status: Filter by status (FINISHED, RUNNING, FAILED).
        max_results: Maximum number of results.

    Returns:
        List of run dictionaries.
    """
    client = get_mlflow_client()

    # Build filter string
    filter_parts = []
    if status:
        filter_parts.append(f"attributes.status = '{status}'")

    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    # Search runs
    if experiment_id:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=max_results,
        )
    else:
        # Get default experiment
        exp = client.get_experiment_by_name(settings.mlflow_experiment_name)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
            )
        else:
            runs = []

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
        }
        for run in runs
    ]


def get_run_details(run_id: str) -> dict[str, Any]:
    """Get detailed information about a run.

    Args:
        run_id: MLFlow run ID.

    Returns:
        Run details dictionary.
    """
    client = get_mlflow_client()
    run = client.get_run(run_id)

    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }


def get_run_metrics(run_id: str) -> pl.DataFrame:
    """Get metrics history for a run as Polars DataFrame.

    Args:
        run_id: MLFlow run ID.

    Returns:
        Polars DataFrame with metrics time series.
    """
    client = get_mlflow_client()
    run = client.get_run(run_id)

    # Get metric history
    metrics_data = []
    for metric_key in run.data.metrics:
        history = client.get_metric_history(run_id, metric_key)
        for metric in history:
            metrics_data.append(
                {
                    "step": metric.step,
                    "metric": metric_key,
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                }
            )

    if not metrics_data:
        return pl.DataFrame(
            schema={"step": pl.Int64, "metric": pl.Utf8, "value": pl.Float64, "timestamp": pl.Int64}
        )

    return pl.DataFrame(metrics_data)


def list_artifacts(run_id: str, path: str = "") -> list[dict[str, Any]]:
    """List artifacts for a run.

    Args:
        run_id: MLFlow run ID.
        path: Artifact path to list.

    Returns:
        List of artifact metadata.
    """
    client = get_mlflow_client()
    artifacts = client.list_artifacts(run_id, path=path)

    return [
        {
            "path": artifact.path,
            "is_dir": artifact.is_dir,
            "file_size": artifact.file_size,
        }
        for artifact in artifacts
    ]


def get_model_versions(model_name: str) -> list[dict[str, Any]]:
    """Get versions of a registered model.

    Args:
        model_name: Model name.

    Returns:
        List of model version dictionaries.
    """
    client = get_mlflow_client()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "name": mv.name,
                "version": mv.version,
                "creation_timestamp": mv.creation_timestamp,
                "last_updated_timestamp": mv.last_updated_timestamp,
                "current_stage": mv.current_stage,
                "description": mv.description,
                "source": mv.source,
                "run_id": mv.run_id,
            }
            for mv in versions
        ]
    except Exception:
        return []
