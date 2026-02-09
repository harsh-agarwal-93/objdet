"""System API endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import settings

router = APIRouter()


@router.get("/status")
def get_system_status() -> dict[str, dict]:
    """Get system status and service health.

    Returns:
        System status information.
    """
    # Check Celery connectivity
    celery_status = "unknown"
    try:
        from backend.services.celery_service import celery_app

        # Use ping() which is faster and more reliable than stats() for simple health check
        # We set a short timeout to avoid blocking
        ping_result = celery_app.control.ping(timeout=1.0)

        # ping returns a list of dictionaries {worker_name: response}
        # If we have at least one worker responding, we are connected
        if ping_result:
            celery_status = "connected"
        else:
            celery_status = "disconnected"
    except Exception as e:
        import traceback

        print(f"Celery status check failed: {e}")
        traceback.print_exc()
        celery_status = "error"

    # Check MLFlow connectivity
    mlflow_status = "unknown"
    try:
        # Use the /health endpoint via requests to avoid issues with the MLFlow client libraries
        # enforcing strict Host header validation or other complex logic.
        # This is a simple connectivity check.
        import requests

        # Extract base URL from tracking URI
        tracking_uri = settings.mlflow_tracking_uri.rstrip("/")
        health_url = f"{tracking_uri}/health"

        response = requests.get(health_url, timeout=2.0)
        if response.status_code == 200:
            mlflow_status = "connected"
        else:
            print(f"MLFlow returned status {response.status_code}")
            mlflow_status = "error"
    except Exception as e:
        print(f"MLFlow status check failed: {e}")
        mlflow_status = "error"

    return {
        "services": {
            "celery": {
                "status": celery_status,
                "broker_url": settings.celery_broker_url,
            },
            "mlflow": {
                "status": mlflow_status,
                "tracking_uri": settings.mlflow_tracking_uri,
            },
        },
        "api": {
            "version": settings.api_version,
            "title": settings.api_title,
        },
    }


@router.get("/config")
def get_system_config() -> dict[str, dict]:
    """Get non-sensitive system configuration.

    Returns:
        System configuration.
    """
    return {
        "celery": {
            "broker_url": settings.celery_broker_url,
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment_name": settings.mlflow_experiment_name,
        },
        "api": {
            "title": settings.api_title,
            "version": settings.api_version,
        },
    }
