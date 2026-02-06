"""Celery-based job pipeline infrastructure.

This package provides:
- Celery task definitions for training jobs
- Resource-based routing to appropriate workers
- Job dependencies and DAG workflows
- Job submission SDK
"""

from objdet.pipelines.celery_app import app
from objdet.pipelines.job import Job, JobStatus
from objdet.pipelines.sdk import get_job_status, submit_job
from objdet.pipelines.tasks import export_model, preprocess_data, train_model

__all__ = [
    "Job",
    "JobStatus",
    "app",
    "export_model",
    "get_job_status",
    "preprocess_data",
    "submit_job",
    "train_model",
]
