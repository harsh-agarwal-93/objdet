"""Celery-based job pipeline infrastructure.

This package provides:
- Celery task definitions for training jobs
- Resource-based routing to appropriate workers
- Job dependencies and DAG workflows
- Job submission SDK
"""

from objdet.pipelines.celery_app import app
from objdet.pipelines.tasks import train_model, export_model, preprocess_data
from objdet.pipelines.job import Job, JobStatus
from objdet.pipelines.sdk import submit_job, get_job_status

__all__ = [
    "app",
    "train_model",
    "export_model",
    "preprocess_data",
    "Job",
    "JobStatus",
    "submit_job",
    "get_job_status",
]
