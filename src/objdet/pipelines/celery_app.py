"""Celery application configuration.

This module configures the Celery app for distributed task execution
with RabbitMQ as the message broker.

Configuration can be provided via environment variables:
- CELERY_BROKER_URL: RabbitMQ URL (default: amqp://guest:guest@localhost:5672//)
- CELERY_RESULT_BACKEND: Result backend URL (default: rpc://)

Example:
    >>> from objdet.pipelines.celery_app import app
    >>>
    >>> # Run worker
    >>> # celery -A objdet.pipelines.celery_app worker --loglevel=info
"""

from __future__ import annotations

import os
from typing import Any

from celery import Celery

from objdet.core.logging import get_logger

logger = get_logger(__name__)

# Default configuration
BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

# Create Celery app
app = Celery(
    "objdet",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["objdet.pipelines.tasks"],
)

# Configure Celery
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=86400,  # 24 hours max
    task_soft_time_limit=82800,  # 23 hours soft limit
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU tasks
    worker_concurrency=1,  # One task per worker for GPU isolation
    # Result settings
    result_expires=86400 * 7,  # Keep results for 7 days
    # Queue routing
    task_routes={
        "objdet.pipelines.tasks.train_model": {"queue": "gpu"},
        "objdet.pipelines.tasks.export_model": {"queue": "gpu"},
        "objdet.pipelines.tasks.preprocess_data": {"queue": "cpu"},
    },
    # Define queues
    task_queues={
        "gpu": {
            "exchange": "gpu",
            "routing_key": "gpu",
        },
        "gpu_high_memory": {
            "exchange": "gpu_high_memory",
            "routing_key": "gpu_high_memory",
        },
        "cpu": {
            "exchange": "cpu",
            "routing_key": "cpu",
        },
    },
)


def get_app() -> Celery:
    """Get the configured Celery app.

    Returns:
        Configured Celery application.
    """
    return app


# Resource-based queue mapping
RESOURCE_QUEUES = {
    "gpu": "gpu",
    "gpu_small": "gpu",
    "gpu_large": "gpu_high_memory",
    "cpu": "cpu",
}


def get_queue_for_resources(
    gpu_count: int = 0,
    gpu_memory_gb: float = 0,
    cpu_only: bool = False,
) -> str:
    """Determine appropriate queue based on resource requirements.

    Args:
        gpu_count: Number of GPUs required.
        gpu_memory_gb: GPU memory required per GPU in GB.
        cpu_only: Whether to run on CPU only.

    Returns:
        Queue name for routing.
    """
    if cpu_only or gpu_count == 0:
        return "cpu"

    if gpu_memory_gb > 24:  # High memory GPU needed (e.g., A100)
        return "gpu_high_memory"

    return "gpu"
