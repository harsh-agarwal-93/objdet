"""MLFlow API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.services import mlflow_service

router = APIRouter()


@router.get("/experiments")
def list_experiments() -> dict[str, list[dict[str, Any]]]:
    """List all MLFlow experiments.

    Returns:
        List of experiments.
    """
    try:
        experiments = mlflow_service.list_experiments()
        return {"experiments": experiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {e!s}")


@router.get("/runs")
def list_runs(
    experiment_id: str | None = Query(None, description="Filter by experiment ID"),
    status: str | None = Query(None, description="Filter by status"),
    max_results: int = Query(100, ge=1, le=1000, description="Maximum results"),
) -> dict[str, list[dict[str, Any]]]:
    """List MLFlow runs with optional filtering.

    Args:
        experiment_id: Filter by experiment ID.
        status: Filter by status.
        max_results: Maximum number of results.

    Returns:
        List of runs.
    """
    try:
        runs = mlflow_service.list_runs(
            experiment_id=experiment_id,
            status=status,
            max_results=max_results,
        )
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {e!s}")


@router.get("/runs/{run_id}")
def get_run_details(run_id: str) -> dict[str, Any]:
    """Get detailed information about a run.

    Args:
        run_id: MLFlow run ID.

    Returns:
        Run details.
    """
    try:
        details = mlflow_service.get_run_details(run_id)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run details: {e!s}")


@router.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> dict[str, Any]:
    """Get metrics time series for a run.

    Args:
        run_id: MLFlow run ID.

    Returns:
        Metrics data as dictionary.
    """
    try:
        df = mlflow_service.get_run_metrics(run_id)
        # Convert Polars DataFrame to dictionary for JSON response
        return {"metrics": df.to_dicts()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get run metrics: {e!s}")


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(
    run_id: str,
    path: str = Query("", description="Artifact path to list"),
) -> dict[str, list[dict[str, Any]]]:
    """List artifacts for a run.

    Args:
        run_id: MLFlow run ID.
        path: Artifact path.

    Returns:
        List of artifacts.
    """
    try:
        artifacts = mlflow_service.list_artifacts(run_id, path=path)
        return {"artifacts": artifacts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list artifacts: {e!s}")


@router.get("/models/{model_name}/versions")
def get_model_versions(model_name: str) -> dict[str, list[dict[str, Any]]]:
    """Get versions of a registered model.

    Args:
        model_name: Model name.

    Returns:
        List of model versions.
    """
    try:
        versions = mlflow_service.get_model_versions(model_name)
        return {"versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model versions: {e!s}")
