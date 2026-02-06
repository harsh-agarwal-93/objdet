"""Unit tests for Pipeline SDK."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from objdet.pipelines.job import JobStatus, JobType


class TestPipelineSDK:
    """Tests for pipeline SDK functions."""

    @pytest.fixture(autouse=True)
    def reset_job_store(self) -> None:
        """Reset the in-memory job store before each test."""
        from objdet.pipelines import sdk

        sdk._job_store.clear()

    def test_submit_job_without_dependencies(self) -> None:
        """Test submitting a job without dependencies."""
        from objdet.pipelines.sdk import submit_job

        with patch("objdet.pipelines.sdk._submit_task") as mock_submit:
            job_id = submit_job(
                job_type="train",
                gpu_count=1,
                config_path="configs/test.yaml",
                output_dir="/outputs/test",
            )

            assert job_id is not None
            assert isinstance(job_id, str)
            mock_submit.assert_called_once()

    def test_submit_job_with_dependencies(self) -> None:
        """Test submitting a job with dependencies."""
        from objdet.pipelines.sdk import submit_job

        with (
            patch("objdet.pipelines.sdk._submit_task"),
            patch("objdet.pipelines.sdk._wait_and_submit") as mock_wait,
        ):
            # First job (no deps)
            job1_id = submit_job(job_type="preprocess", input_dir="/data")

            # Second job depends on first
            job2_id = submit_job(
                job_type="train",
                dependencies=[job1_id],
                config_path="configs/test.yaml",
            )

            assert job2_id is not None
            mock_wait.assert_called_once()

    def test_submit_job_with_tags(self) -> None:
        """Test submitting a job with tags."""
        from objdet.pipelines.sdk import _job_store, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            job_id = submit_job(
                job_type="train",
                tags=["experiment", "v1"],
                config_path="configs/test.yaml",
            )

            job = _job_store[job_id]
            assert "experiment" in job.tags
            assert "v1" in job.tags

    def test_get_job_status_found(self) -> None:
        """Test getting status of existing job."""
        from objdet.pipelines.sdk import get_job_status, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            job_id = submit_job(job_type="export", checkpoint_path="/model.ckpt")

            status = get_job_status(job_id)

            assert "id" in status
            assert status["id"] == job_id
            assert "status" in status

    def test_get_job_status_not_found(self) -> None:
        """Test getting status of non-existent job."""
        from objdet.pipelines.sdk import get_job_status

        status = get_job_status("nonexistent-job-id")

        assert "error" in status
        assert "not found" in status["error"]

    def test_cancel_job(self) -> None:
        """Test cancelling a job."""
        from objdet.pipelines.sdk import _job_store, cancel_job, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            job_id = submit_job(job_type="train", config_path="test.yaml")

            result = cancel_job(job_id)

            assert result is True
            assert _job_store[job_id].status == JobStatus.CANCELLED

    def test_cancel_job_not_found(self) -> None:
        """Test cancelling non-existent job."""
        from objdet.pipelines.sdk import cancel_job

        result = cancel_job("nonexistent-job-id")

        assert result is False

    def test_list_jobs_no_filters(self) -> None:
        """Test listing all jobs without filters."""
        from objdet.pipelines.sdk import list_jobs, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            submit_job(job_type="train", config_path="test1.yaml")
            submit_job(job_type="export", checkpoint_path="model.ckpt")
            submit_job(job_type="preprocess", input_dir="/data")

            jobs = list_jobs()

            assert len(jobs) == 3

    def test_list_jobs_with_type_filter(self) -> None:
        """Test listing jobs filtered by type."""
        from objdet.pipelines.sdk import list_jobs, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            submit_job(job_type="train", config_path="test1.yaml")
            submit_job(job_type="train", config_path="test2.yaml")
            submit_job(job_type="export", checkpoint_path="model.ckpt")

            jobs = list_jobs(job_type=JobType.TRAIN)

            assert len(jobs) == 2
            assert all(j["job_type"] == "train" for j in jobs)

    def test_list_jobs_with_tags_filter(self) -> None:
        """Test listing jobs filtered by tags."""
        from objdet.pipelines.sdk import list_jobs, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            submit_job(job_type="train", tags=["experiment"], config_path="test1.yaml")
            submit_job(job_type="train", tags=["production"], config_path="test2.yaml")

            jobs = list_jobs(tags=["experiment"])

            assert len(jobs) == 1
            assert "experiment" in jobs[0]["tags"]

    def test_submit_job_priority(self) -> None:
        """Test that priority is set correctly."""
        from objdet.pipelines.sdk import _job_store, submit_job

        with patch("objdet.pipelines.sdk._submit_task"):
            job_id = submit_job(
                job_type="train",
                priority=10,
                config_path="test.yaml",
            )

            job = _job_store[job_id]
            assert job.priority == 10
