"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import mlflow, system, training
from backend.core.config import settings

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    docs_url=settings.api_docs_url,
    redoc_url=settings.api_redoc_url,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(mlflow.router, prefix="/api/mlflow", tags=["mlflow"])
app.include_router(system.router, prefix="/api/system", tags=["system"])


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status message indicating service health.
    """
    return {"status": "healthy"}


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint with API information.

    Returns:
        API information and documentation links.
    """
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": settings.api_docs_url,
    }
