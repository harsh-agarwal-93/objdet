"""Job model and status tracking.

This module defines the Job model for tracking task execution
status and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from whenever import Instant


class JobStatus(str, Enum):
    """Status of a pipeline job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobType(str, Enum):
    """Type of pipeline job."""

    TRAIN = "train"
    EXPORT = "export"
    PREPROCESS = "preprocess"
    EVALUATE = "evaluate"


@dataclass
class Job:
    """Represents a pipeline job.

    Attributes:
        id: Unique job identifier.
        job_type: Type of job.
        status: Current status.
        config: Job configuration.
        dependencies: IDs of jobs this depends on.
        created_at: Creation timestamp.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        result: Job result data.
        error: Error message if failed.
        celery_task_id: Celery task ID.
    """

    job_type: JobType
    config: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    created_at: Instant = field(default_factory=Instant.now)
    started_at: Instant | None = None
    completed_at: Instant | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    celery_task_id: str | None = None
    priority: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "config": self.config,
            "dependencies": self.dependencies,
            "created_at": str(self.created_at),
            "started_at": str(self.started_at) if self.started_at else None,
            "completed_at": str(self.completed_at) if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "celery_task_id": self.celery_task_id,
            "priority": self.priority,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        """Create job from dictionary."""
        job = cls(
            id=data["id"],
            job_type=JobType(data["job_type"]),
            status=JobStatus(data["status"]),
            config=data["config"],
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", 0),
            tags=data.get("tags", []),
        )

        if data.get("created_at"):
            job.created_at = Instant.parse_common_iso(data["created_at"])
        if data.get("started_at"):
            job.started_at = Instant.parse_common_iso(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = Instant.parse_common_iso(data["completed_at"])

        job.result = data.get("result")
        job.error = data.get("error")
        job.celery_task_id = data.get("celery_task_id")

        return job

    def is_ready(self, completed_jobs: set[str]) -> bool:
        """Check if all dependencies are satisfied.

        Args:
            completed_jobs: Set of completed job IDs.

        Returns:
            True if job can be executed.
        """
        return all(dep in completed_jobs for dep in self.dependencies)


@dataclass
class JobDAG:
    """Directed Acyclic Graph of jobs with dependencies.

    Manages job execution order based on dependencies.
    """

    jobs: dict[str, Job] = field(default_factory=dict)

    def add_job(self, job: Job) -> None:
        """Add a job to the DAG."""
        # Validate no circular dependencies
        self._validate_no_cycles(job)
        self.jobs[job.id] = job

    def _validate_no_cycles(self, new_job: Job) -> None:
        """Validate adding job doesn't create cycles."""
        # Simple DFS to detect cycles
        visited = set()

        def dfs(job_id: str) -> bool:
            if job_id == new_job.id:
                return True  # Cycle found

            if job_id in visited:
                return False

            visited.add(job_id)

            if job_id in self.jobs:
                for dep_id in self.jobs[job_id].dependencies:
                    if dfs(dep_id):
                        return True

            return False

        for dep_id in new_job.dependencies:
            if dfs(dep_id):
                raise ValueError(f"Adding job would create cycle: {new_job.id}")

    def get_ready_jobs(self) -> list[Job]:
        """Get jobs that are ready to execute.

        Returns:
            List of jobs with all dependencies satisfied.
        """
        completed = {jid for jid, job in self.jobs.items() if job.status == JobStatus.COMPLETED}

        return [
            job
            for job in self.jobs.values()
            if job.status == JobStatus.PENDING and job.is_ready(completed)
        ]

    def get_execution_order(self) -> list[str]:
        """Get topological order for job execution.

        Returns:
            List of job IDs in execution order.
        """
        # Kahn's algorithm
        in_degree = dict.fromkeys(self.jobs, 0)
        for job in self.jobs.values():
            for dep_id in job.dependencies:
                if dep_id in in_degree:
                    in_degree[job.id] += 1

        queue = [jid for jid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            jid = queue.pop(0)
            order.append(jid)

            for job in self.jobs.values():
                if jid in job.dependencies:
                    in_degree[job.id] -= 1
                    if in_degree[job.id] == 0:
                        queue.append(job.id)

        return order
