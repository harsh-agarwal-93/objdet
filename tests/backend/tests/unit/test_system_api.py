"""Unit tests for system API endpoints and main application."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_health_check(test_client: TestClient) -> None:
    """Test health check endpoint."""
    response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(test_client: TestClient) -> None:
    """Test root endpoint."""
    response = test_client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data
    assert data["name"] == "ObjDet WebApp API"


def test_cors_headers(test_client: TestClient) -> None:
    """Test CORS middleware configuration."""
    response = test_client.options(
        "/api/mlflow/experiments",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    # Verify CORS headers are present
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers


def test_cors_required_origins() -> None:
    """Test that Docker-required origins are in CORS config."""
    from backend.core.config import settings

    required_origins = [
        "http://localhost:3000",  # React frontend (dev)
    ]
    for origin in required_origins:
        assert origin in settings.allowed_origins, f"Missing required CORS origin: {origin}"


def test_api_docs_available(test_client: TestClient) -> None:
    """Test that API documentation is available."""
    response = test_client.get("/api/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema(test_client: TestClient) -> None:
    """Test that OpenAPI schema is available."""
    response = test_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema

    # Verify our endpoints are in the schema
    assert "/api/training/submit" in schema["paths"]
    assert "/api/mlflow/experiments" in schema["paths"]
    assert "/health" in schema["paths"]


def test_get_system_status(test_client: TestClient) -> None:
    """Test system status endpoint returns service health."""
    response = test_client.get("/api/system/status")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "services" in data
    assert "api" in data
    assert "celery" in data["services"]
    assert "mlflow" in data["services"]

    # Verify service info
    assert "status" in data["services"]["celery"]
    assert "status" in data["services"]["mlflow"]


def test_get_system_config(test_client: TestClient) -> None:
    """Test system config endpoint returns configuration."""
    response = test_client.get("/api/system/config")

    assert response.status_code == 200
    data = response.json()

    # Verify config structure
    assert "celery" in data
    assert "mlflow" in data
    assert "api" in data

    # Verify celery config
    assert "broker_url" in data["celery"]

    # Verify mlflow config
    assert "tracking_uri" in data["mlflow"]
    assert "experiment_name" in data["mlflow"]

    # Verify API config
    assert "title" in data["api"]
    assert "version" in data["api"]


def test_system_status_mlflow_error(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test system status when MLFlow connection fails.

    Args:
        test_client: FastAPI test client.
        monkeypatch: Pytest monkeypatch fixture.
    """
    from unittest.mock import Mock

    # Mock get_mlflow_client to return a client that raises error
    mock_client = Mock()
    mock_client.search_experiments.side_effect = Exception("MLFlow unavailable")

    # We must patch get_mlflow_client because it is already patched by autouse fixture
    monkeypatch.setattr(
        "backend.services.mlflow_service.get_mlflow_client",
        lambda tracking_uri=None: mock_client,
        raising=False,
    )

    response = test_client.get("/api/system/status")

    assert response.status_code == 200
    data = response.json()

    # Should handle error gracefully
    assert "services" in data
    assert data["services"]["mlflow"]["status"] == "error"


def test_system_status_celery_error(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test system status when Celery connection fails.

    Args:
        test_client: FastAPI test client.
        monkeypatch: Pytest monkeypatch fixture.
    """
    from unittest.mock import Mock

    # Mock celery control inspect
    mock_inspect = Mock()
    mock_inspect.stats.side_effect = Exception("Celery unavailable")

    mock_control = Mock()
    mock_control.inspect.return_value = mock_inspect

    # Patch the control object on the app
    # Note: control is accessed via celery_app.control
    monkeypatch.setattr(
        "backend.services.celery_service.celery_app.control",
        mock_control,
        raising=False,
    )

    response = test_client.get("/api/system/status")

    assert response.status_code == 200
    data = response.json()

    # Should handle error gracefully
    assert "services" in data
    assert data["services"]["celery"]["status"] == "error"


def test_concurrent_status_requests(test_client: TestClient) -> None:
    """Test multiple simultaneous status requests.

    Args:
        test_client: FastAPI test client.
    """
    import concurrent.futures

    def make_request() -> Any:
        return test_client.get("/api/system/status")

    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]

    # All should succeed
    assert all(r.status_code == 200 for r in responses)

    # All should have valid structure
    for response in responses:
        data = response.json()
        assert "services" in data
        assert "api" in data


def test_config_endpoint_no_secrets(test_client: TestClient) -> None:
    """Test config endpoint doesn't expose sensitive information.

    Args:
        test_client: FastAPI test client.
    """
    response = test_client.get("/api/system/config")

    assert response.status_code == 200
    data = response.json()

    # Verify only non-sensitive config is returned
    assert "celery" in data
    assert "mlflow" in data

    # Should not contain sensitive fields (if any exist in settings)
    data_str = str(data).lower()
    assert "password" not in data_str
    assert "secret" not in data_str
