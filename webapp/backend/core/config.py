"""Backend configuration settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Celery Configuration
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "rpc://"

    # MLFlow Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "objdet"

    # API Configuration
    api_title: str = "ObjDet WebApp API"
    api_version: str = "0.1.0"
    api_docs_url: str = "/api/docs"
    api_redoc_url: str = "/api/redoc"

    # CORS Configuration
    allowed_origins: list[str] = ["http://localhost:8501"]


settings = Settings()
