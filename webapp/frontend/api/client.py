"""HTTP client for FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx
import polars as pl


class BackendClient:
    """Client for interacting with FastAPI backend."""

    def __init__(self, base_url: str | None = None):
        """Initialize backend client.

        Args:
            base_url: Backend URL (defaults to env var or localhost).
        """
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()

    # Training endpoints
    def submit_training_job(self, config: dict[str, Any]) -> dict[str, Any]:
        """Submit training job to backend.

        Args:
            config: Training configuration dictionary.

        Returns:
            Training job response with task ID.
        """
        response = self.client.post("/api/training/submit", json=config)
        response.raise_for_status()
        return response.json()

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get status of a training task.

        Args:
            task_id: Celery task ID.

        Returns:
            Task status information.
        """
        response = self.client.get(f"/api/training/status/{task_id}")
        response.raise_for_status()
        return response.json()

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a running task.

        Args:
            task_id: Celery task ID.

        Returns:
            Cancellation confirmation.
        """
        response = self.client.post(f"/api/training/cancel/{task_id}")
        response.raise_for_status()
        return response.json()

    def list_active_tasks(self) -> list[dict[str, Any]]:
        """List all active training tasks.

        Returns:
            List of active tasks.
        """
        response = self.client.get("/api/training/active")
        response.raise_for_status()
        return response.json().get("tasks", [])

    # MLFlow endpoints
    def list_experiments(self) -> list[dict[str, Any]]:
        """List all MLFlow experiments.

        Returns:
            List of experiments.
        """
        response = self.client.get("/api/mlflow/experiments")
        response.raise_for_status()
        return response.json().get("experiments", [])

    def list_runs(
        self,
        experiment_id: str | None = None,
        status: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """List MLFlow runs with filtering.

        Args:
            experiment_id: Filter by experiment ID.
            status: Filter by status.
            max_results: Maximum results.

        Returns:
            List of runs.
        """
        params = {}
        if experiment_id:
            params["experiment_id"] = experiment_id
        if status:
            params["status"] = status
        params["max_results"] = max_results

        response = self.client.get("/api/mlflow/runs", params=params)
        response.raise_for_status()
        return response.json().get("runs", [])

    def get_run_details(self, run_id: str) -> dict[str, Any]:
        """Get detailed run information.

        Args:
            run_id: MLFlow run ID.

        Returns:
            Run details.
        """
        response = self.client.get(f"/api/mlflow/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    def get_run_metrics(self, run_id: str) -> pl.DataFrame:
        """Get run metrics as Polars DataFrame.

        Args:
            run_id: MLFlow run ID.

        Returns:
            Polars DataFrame with metrics.
        """
        response = self.client.get(f"/api/mlflow/runs/{run_id}/metrics")
        response.raise_for_status()
        metrics_data = response.json().get("metrics", [])

        if not metrics_data:
            return pl.DataFrame(schema={"step": pl.Int64, "metric": pl.Utf8, "value": pl.Float64})

        return pl.DataFrame(metrics_data)

    def list_artifacts(self, run_id: str, path: str = "") -> list[dict[str, Any]]:
        """List artifacts for a run.

        Args:
            run_id: MLFlow run ID.
            path: Artifact path.

        Returns:
            List of artifacts.
        """
        params = {"path": path} if path else {}
        response = self.client.get(f"/api/mlflow/runs/{run_id}/artifacts", params=params)
        response.raise_for_status()
        return response.json().get("artifacts", [])

    # System endpoints
    def get_system_status(self) -> dict[str, Any]:
        """Get system status.

        Returns:
            System status information.
        """
        response = self.client.get("/api/system/status")
        response.raise_for_status()
        return response.json()


# Singleton instance
_client: BackendClient | None = None


def get_client() -> BackendClient:
    """Get or create backend client singleton.

    Returns:
        Backend client instance.
    """
    global _client
    if _client is None:
        _client = BackendClient()
    return _client
