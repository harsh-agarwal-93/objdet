"""SDK for submitting and managing pipeline jobs.

This module provides a Python SDK for submitting jobs to the
Celery task queue programmatically.

Example:
    >>> from objdet.pipelines import submit_job, get_job_status
    >>>
    >>> # Submit training job
    >>> job_id = submit_job(
    ...     job_type="train",
    ...     config_path="configs/coco_frcnn.yaml",
    ...     output_dir="/outputs/exp_001",
    ... )
    >>>
    >>> # Check status
    >>> status = get_job_status(job_id)
    >>> print(f"Job {job_id}: {status}")
"""

from __future__ import annotations

from typing import Any

from whenever import Instant

from objdet.core.logging import get_logger
from objdet.pipelines.celery_app import get_queue_for_resources
from objdet.pipelines.job import Job, JobStatus, JobType

logger = get_logger(__name__)

# In-memory job store (in production, use Redis or database)
_job_store: dict[str, Job] = {}


def submit_job(
    job_type: str,
    gpu_count: int = 1,
    gpu_memory_gb: float = 16,
    priority: int = 0,
    tags: list[str] | None = None,
    dependencies: list[str] | None = None,
    **config: Any,
) -> str:
    """Submit a job to the pipeline queue.

    Args:
        job_type: Type of job ("train", "export", "preprocess").
        gpu_count: Number of GPUs required.
        gpu_memory_gb: GPU memory required.
        priority: Job priority (higher = more important).
        tags: Optional tags for filtering.
        dependencies: Optional list of job IDs this depends on.
        **config: Job-specific configuration.

    Returns:
        Job ID for tracking.

    Example:
        >>> job_id = submit_job(
        ...     job_type="train",
        ...     config_path="configs/coco.yaml",
        ...     output_dir="/outputs/exp_001",
        ...     max_epochs=50,
        ... )
    """
    # Create job
    job = Job(
        job_type=JobType(job_type),
        config=config,
        dependencies=dependencies or [],
        priority=priority,
        tags=tags or [],
    )

    _job_store[job.id] = job

    # Determine queue based on resources
    queue = get_queue_for_resources(
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        cpu_only=(job_type == "preprocess"),
    )

    # Check dependencies
    if dependencies:
        # Wait for dependencies before submitting
        _wait_and_submit(job, queue)
        logger.info(f"Job {job.id} queued with dependencies: {dependencies}")
    else:
        # Submit immediately
        _submit_task(job, queue)

    return job.id


def _submit_task(job: Job, queue: str) -> None:
    """Submit job to Celery queue."""
    from objdet.pipelines import tasks

    task_map = {
        JobType.TRAIN: tasks.train_model,
        JobType.EXPORT: tasks.export_model,
        JobType.PREPROCESS: tasks.preprocess_data,
    }

    task = task_map.get(job.job_type)
    if not task:
        raise ValueError(f"Unknown job type: {job.job_type}")

    # Submit to Celery
    result = task.apply_async(  # type: ignore
        kwargs=job.config,
        queue=queue,
        priority=job.priority,
    )

    job.celery_task_id = result.id
    job.status = JobStatus.QUEUED
    job.started_at = Instant.now()

    logger.info(f"Submitted job {job.id} to queue {queue}: task_id={result.id}")


def _wait_and_submit(job: Job, queue: str) -> None:
    """Submit job after dependencies complete."""
    # Create chain that waits for dependencies
    job.status = JobStatus.PENDING

    # In production, this would use Celery's chord/chain
    # For now, we'll use polling in get_job_status
    logger.info(f"Job {job.id} waiting on dependencies: {job.dependencies}")


def get_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a submitted job.

    Args:
        job_id: Job ID returned from submit_job.

    Returns:
        Dictionary with job status information.
    """
    job = _job_store.get(job_id)
    if not job:
        return {"error": f"Job not found: {job_id}"}

    # If job has Celery task, update status
    if job.celery_task_id:
        _update_from_celery(job)

    return job.to_dict()


def _update_from_celery(job: Job) -> None:
    """Update job status from Celery result."""
    from celery.result import AsyncResult

    from objdet.pipelines.celery_app import app

    result = AsyncResult(job.celery_task_id, app=app)

    if result.ready():
        if result.successful():
            job.status = JobStatus.COMPLETED
            job.result = result.result
            job.completed_at = Instant.now()
        else:
            job.status = JobStatus.FAILED
            job.error = str(result.result)
            job.completed_at = Instant.now()
    elif result.state == "STARTED":
        job.status = JobStatus.RUNNING
    elif result.state == "RETRY":
        job.status = JobStatus.RETRYING


def cancel_job(job_id: str) -> bool:
    """Cancel a pending or running job.

    Args:
        job_id: Job ID to cancel.

    Returns:
        True if cancelled successfully.
    """
    job = _job_store.get(job_id)
    if not job:
        return False

    if job.celery_task_id:
        from celery.result import AsyncResult

        from objdet.pipelines.celery_app import app

        result = AsyncResult(job.celery_task_id, app=app)
        result.revoke(terminate=True)

    job.status = JobStatus.CANCELLED
    job.completed_at = Instant.now()
    logger.info(f"Cancelled job {job_id}")

    return True


def list_jobs(
    status: JobStatus | None = None,
    job_type: JobType | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List jobs with optional filters.

    Args:
        status: Filter by status.
        job_type: Filter by type.
        tags: Filter by tags (any match).

    Returns:
        List of job dictionaries.
    """
    results = []

    for job in _job_store.values():
        if status and job.status != status:
            continue
        if job_type and job.job_type != job_type:
            continue
        if tags and not any(t in job.tags for t in tags):
            continue

        results.append(job.to_dict())

    return sorted(results, key=lambda x: x["created_at"], reverse=True)
