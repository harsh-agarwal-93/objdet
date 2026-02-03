"""Standalone Celery application for webapp backend."""

from __future__ import annotations

import os

from celery import Celery

# Get broker URL from environment
broker_url = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

# Create Celery app
app = Celery(
    "objdet_webapp",
    broker=broker_url,
    backend=result_backend,
)

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    result_expires=3600,  # Results expire after 1 hour
)

# Auto-discover tasks in the tasks module
app.autodiscover_tasks(["backend"])
