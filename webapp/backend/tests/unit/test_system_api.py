"""Unit tests for system API endpoints and main application."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )

    # Verify CORS headers are present
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers


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
