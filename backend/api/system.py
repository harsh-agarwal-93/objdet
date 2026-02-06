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

        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        celery_status = "connected" if stats else "disconnected"
    except Exception:
        celery_status = "error"

    # Check MLFlow connectivity
    mlflow_status = "unknown"
    try:
        from backend.services.mlflow_service import get_mlflow_client

        client = get_mlflow_client()
        client.search_experiments(max_results=1)
        mlflow_status = "connected"
    except Exception:
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
