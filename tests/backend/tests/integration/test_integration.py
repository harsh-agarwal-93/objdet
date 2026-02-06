"""Integration tests for webapp backend.

These tests require live services (RabbitMQ, MLFlow, Celery) to be running.
Run with: pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from mlflow.client import MlflowClient


# Celery Integration Tests


@pytest.mark.integration
def test_celery_connection(live_celery_app: Any) -> None:
    """Test Celery can connect to RabbitMQ broker.

    Args:
        live_celery_app: Live Celery application fixture.
    """
    # Verify broker connection
    inspect = live_celery_app.control.inspect()
    assert inspect is not None

    # Get active workers (may be empty if worker just started)
    stats = inspect.stats()
    # Stats should be a dict (even if empty) if connection works
    assert isinstance(stats, dict) or stats is None


@pytest.mark.integration
def test_celery_task_status_retrieval(live_celery_app: Any) -> None:
    """Test querying task status from Celery backend.

    Args:
        live_celery_app: Live Celery application fixture.
    """
    from celery.result import AsyncResult

    # Create a dummy task ID
    task_id = "test-dummy-task-123"
    result = AsyncResult(task_id, app=live_celery_app)

    # Should be able to query status (even for non-existent task)
    assert result.state in ["PENDING", "SUCCESS", "FAILURE"]


@pytest.mark.integration
def test_celery_inspect_active_tasks(live_celery_app: Any) -> None:
    """Test inspecting active Celery tasks.

    Args:
        live_celery_app: Live Celery application fixture.
    """
    inspect = live_celery_app.control.inspect()
    active_tasks = inspect.active()

    # Should return dict (may be empty if no workers/tasks)
    assert isinstance(active_tasks, dict) or active_tasks is None


# MLFlow Integration Tests


@pytest.mark.integration
def test_mlflow_connection(live_mlflow_client: MlflowClient) -> None:
    """Test MLFlow tracking server is accessible.

    Args:
        live_mlflow_client: Live MLFlow client fixture.
    """
    # List experiments should work
    experiments = live_mlflow_client.search_experiments()
    assert isinstance(experiments, list)

    # Should have at least the default experiment
    assert len(experiments) >= 1


@pytest.mark.integration
def test_mlflow_create_experiment(live_mlflow_client: MlflowClient) -> None:
    """Test creating a new MLFlow experiment.

    Args:
        live_mlflow_client: Live MLFlow client fixture.
    """
    exp_name = f"test_experiment_{int(time.time())}"

    # Create experiment
    exp_id = live_mlflow_client.create_experiment(exp_name)
    assert exp_id is not None

    # Verify it exists
    exp = live_mlflow_client.get_experiment(exp_id)
    assert exp.name == exp_name
    assert exp.lifecycle_stage == "active"

    # Cleanup
    live_mlflow_client.delete_experiment(exp_id)


@pytest.mark.integration
def test_mlflow_log_run(live_mlflow_client: MlflowClient) -> None:
    """Test logging a complete run with parameters and metrics.

    Args:
        live_mlflow_client: Live MLFlow client fixture.
    """
    exp_name = f"test_run_experiment_{int(time.time())}"
    exp_id = live_mlflow_client.create_experiment(exp_name)

    try:
        # Create a run
        run = live_mlflow_client.create_run(exp_id)
        run_id = run.info.run_id

        # Log parameters
        live_mlflow_client.log_param(run_id, "learning_rate", 0.001)
        live_mlflow_client.log_param(run_id, "batch_size", 32)

        # Log metrics
        live_mlflow_client.log_metric(run_id, "loss", 0.5, step=0)
        live_mlflow_client.log_metric(run_id, "accuracy", 0.85, step=0)

        # End run
        live_mlflow_client.set_terminated(run_id)

        # Verify run data
        run_data = live_mlflow_client.get_run(run_id)
        assert run_data.data.params["learning_rate"] == "0.001"
        assert run_data.data.params["batch_size"] == "32"
        assert "loss" in run_data.data.metrics
        assert "accuracy" in run_data.data.metrics

    finally:
        # Cleanup
        live_mlflow_client.delete_experiment(exp_id)


@pytest.mark.integration
def test_mlflow_query_runs(live_mlflow_client: MlflowClient) -> None:
    """Test querying and retrieving logged runs.

    Args:
        live_mlflow_client: Live MLFlow client fixture.
    """
    exp_name = f"test_query_experiment_{int(time.time())}"
    exp_id = live_mlflow_client.create_experiment(exp_name)

    try:
        # Create multiple runs
        run1 = live_mlflow_client.create_run(exp_id)
        live_mlflow_client.log_metric(run1.info.run_id, "metric1", 1.0)
        live_mlflow_client.set_terminated(run1.info.run_id)

        run2 = live_mlflow_client.create_run(exp_id)
        live_mlflow_client.log_metric(run2.info.run_id, "metric1", 2.0)
        live_mlflow_client.set_terminated(run2.info.run_id)

        # Query runs
        runs = live_mlflow_client.search_runs(experiment_ids=[exp_id])
        assert len(runs) == 2

        # Verify runs have the logged metrics
        metrics = [run.data.metrics.get("metric1") for run in runs]
        assert 1.0 in metrics
        assert 2.0 in metrics

    finally:
        # Cleanup
        live_mlflow_client.delete_experiment(exp_id)


# End-to-End Workflow Tests


@pytest.mark.integration
def test_system_health_check_e2e(test_client: TestClient) -> None:
    """Test system health endpoint reports all services.

    Args:
        test_client: FastAPI test client.
    """
    response = test_client.get("/api/system/status")
    assert response.status_code == 200

    data = response.json()
    assert "services" in data
    assert "celery" in data["services"]
    assert "mlflow" in data["services"]

    # Both services should be connected
    assert data["services"]["celery"]["status"] == "connected"
    assert data["services"]["mlflow"]["status"] == "connected"


@pytest.mark.integration
def test_mlflow_experiments_api_e2e(
    test_client: TestClient, live_mlflow_client: MlflowClient
) -> None:
    """Test MLFlow experiments API with real MLFlow backend.

    Args:
        test_client: FastAPI test client.
        live_mlflow_client: Live MLFlow client.
    """
    # Create a test experiment directly in MLFlow
    exp_name = f"test_api_experiment_{int(time.time())}"
    exp_id = live_mlflow_client.create_experiment(exp_name)

    try:
        # Query via API
        response = test_client.get("/api/mlflow/experiments")
        assert response.status_code == 200

        data = response.json()
        assert "experiments" in data
        experiments = data["experiments"]

        # Find our experiment
        exp_names = [exp["name"] for exp in experiments]
        assert exp_name in exp_names

    finally:
        # Cleanup
        live_mlflow_client.delete_experiment(exp_id)


@pytest.mark.integration
def test_mlflow_runs_api_e2e(test_client: TestClient, live_mlflow_client: MlflowClient) -> None:
    """Test MLFlow runs API retrieves data from real backend.

    Args:
        test_client: FastAPI test client.
        live_mlflow_client: Live MLFlow client.
    """
    # Create experiment and run
    exp_name = f"test_runs_api_{int(time.time())}"
    exp_id = live_mlflow_client.create_experiment(exp_name)

    try:
        # Create a run with some data
        run = live_mlflow_client.create_run(exp_id)
        run_id = run.info.run_id
        live_mlflow_client.log_metric(run_id, "test_metric", 42.0)
        live_mlflow_client.set_terminated(run_id)

        # Query runs via API (using experiment name)
        # First, we need to get experiment by name
        exp = live_mlflow_client.get_experiment_by_name(exp_name)
        assert exp is not None

        # Query runs for this experiment
        response = test_client.get(f"/api/mlflow/runs?experiment_id={exp.experiment_id}")
        assert response.status_code == 200

        data = response.json()
        assert "runs" in data
        runs = data["runs"]
        assert len(runs) >= 1

        # Verify our run is present
        run_ids = [r["run_id"] for r in runs]
        assert run_id in run_ids

    finally:
        # Cleanup
        live_mlflow_client.delete_experiment(exp_id)
