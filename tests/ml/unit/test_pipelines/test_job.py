import pytest
from whenever import Instant

from objdet.pipelines.job import Job, JobDAG, JobStatus, JobType


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        """Test creating a job."""
        job = Job(
            job_type=JobType.TRAIN,
            config={"config_path": "test.yaml"},
        )

        assert job.job_type == JobType.TRAIN
        assert job.status == JobStatus.PENDING
        assert job.id is not None
        assert len(job.dependencies) == 0

    def test_job_to_dict(self):
        """Test job serialization."""
        job = Job(
            job_type=JobType.TRAIN,
            config={"config_path": "test.yaml"},
            tags=["experiment"],
        )

        data = job.to_dict()

        assert data["job_type"] == "train"
        assert data["status"] == "pending"
        assert "tags" in data
        assert "experiment" in data["tags"]

    def test_job_from_dict(self):
        """Test job deserialization."""
        data = {
            "id": "test-id",
            "job_type": "train",
            "status": "completed",
            "config": {},
            "created_at": str(Instant.now()),
        }

        job = Job.from_dict(data)

        assert job.id == "test-id"
        assert job.job_type == JobType.TRAIN
        assert job.status == JobStatus.COMPLETED

    def test_job_is_ready_no_deps(self):
        """Job with no dependencies is always ready."""
        job = Job(job_type=JobType.TRAIN, config={})

        assert job.is_ready(set())

    def test_job_is_ready_with_deps(self):
        """Job with dependencies is ready when deps complete."""
        job = Job(
            job_type=JobType.TRAIN,
            config={},
            dependencies=["job-1", "job-2"],
        )

        assert not job.is_ready({"job-1"})
        assert job.is_ready({"job-1", "job-2"})


class TestJobDAG:
    """Tests for JobDAG class."""

    def test_add_job(self):
        """Test adding jobs to DAG."""
        dag = JobDAG()
        job = Job(job_type=JobType.TRAIN, config={})

        dag.add_job(job)

        assert job.id in dag.jobs

    def test_get_ready_jobs(self):
        """Test getting jobs ready to execute."""
        dag = JobDAG()

        # Add two independent jobs
        job1 = Job(job_type=JobType.TRAIN, config={})
        job2 = Job(job_type=JobType.TRAIN, config={})

        dag.add_job(job1)
        dag.add_job(job2)

        ready = dag.get_ready_jobs()

        assert len(ready) == 2

    def test_get_ready_jobs_with_deps(self):
        """Jobs with unsatisfied deps should not be ready."""
        dag = JobDAG()

        job1 = Job(job_type=JobType.TRAIN, config={})
        job2 = Job(
            job_type=JobType.EXPORT,
            config={},
            dependencies=[job1.id],
        )

        dag.add_job(job1)
        dag.add_job(job2)

        ready = dag.get_ready_jobs()

        # Only job1 should be ready
        assert len(ready) == 1
        assert ready[0].id == job1.id

    def test_execution_order(self):
        """Test topological sort order."""
        dag = JobDAG()

        job1 = Job(job_type=JobType.PREPROCESS, config={})
        job2 = Job(
            job_type=JobType.TRAIN,
            config={},
            dependencies=[job1.id],
        )
        job3 = Job(
            job_type=JobType.EXPORT,
            config={},
            dependencies=[job2.id],
        )

        dag.add_job(job1)
        dag.add_job(job2)
        dag.add_job(job3)

        order = dag.get_execution_order()

        # job1 should come before job2, job2 before job3
        assert order.index(job1.id) < order.index(job2.id)
        assert order.index(job2.id) < order.index(job3.id)

    def test_add_job_cycle(self):
        """Test cycle detection."""
        dag = JobDAG()

        # job1 depends on job2
        job1 = Job(job_type=JobType.TRAIN, config={}, dependencies=["job-2"], id="job-1")

        # job2 depends on job1
        job2 = Job(job_type=JobType.TRAIN, config={}, dependencies=["job-1"], id="job-2")

        # Add job1 (job2 not in DAG yet, so no checks on it)
        dag.add_job(job1)

        # Add job2 (should detect cycle: job2 -> job1 -> job2)
        with pytest.raises(ValueError, match="Adding job would create cycle"):
            dag.add_job(job2)
