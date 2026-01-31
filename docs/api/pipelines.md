# Pipelines API

API reference for job pipelines and task queue.

## SDK Functions

### submit_job

```{eval-rst}
.. autofunction:: objdet.pipelines.sdk.submit_job
```

```python
from objdet.pipelines import submit_job

job_id = submit_job(
    job_type="train",
    config_path="configs/coco_frcnn.yaml",
    output_dir="/outputs/exp_001",
    gpu_count=2,
    gpu_memory_gb=16,
    priority=5,
    tags=["experiment", "coco"],
)
```

---

### get_job_status

```{eval-rst}
.. autofunction:: objdet.pipelines.sdk.get_job_status
```

```python
from objdet.pipelines import get_job_status

status = get_job_status(job_id)
print(f"Job {job_id}: {status['status']}")
```

---

### cancel_job

```{eval-rst}
.. autofunction:: objdet.pipelines.sdk.cancel_job
```

```python
from objdet.pipelines import cancel_job

success = cancel_job(job_id)
```

---

### list_jobs

```{eval-rst}
.. autofunction:: objdet.pipelines.sdk.list_jobs
```

```python
from objdet.pipelines import list_jobs
from objdet.pipelines.job import JobStatus, JobType

# List all running training jobs
jobs = list_jobs(status=JobStatus.RUNNING, job_type=JobType.TRAIN)
```

---

## Job Models

### JobStatus

```{eval-rst}
.. autoclass:: objdet.pipelines.job.JobStatus
   :members:
   :undoc-members:
```

**Values:**

- `PENDING` - Job created but not yet queued
- `QUEUED` - Job submitted to Celery queue
- `RUNNING` - Job currently executing
- `COMPLETED` - Job finished successfully
- `FAILED` - Job failed with error
- `CANCELLED` - Job was cancelled
- `RETRYING` - Job is being retried after failure

---

### JobType

```{eval-rst}
.. autoclass:: objdet.pipelines.job.JobType
   :members:
   :undoc-members:
```

**Values:**

- `TRAIN` - Training job
- `EXPORT` - Model export job
- `PREPROCESS` - Data preprocessing job
- `EVALUATE` - Model evaluation job

---

### Job

```{eval-rst}
.. autoclass:: objdet.pipelines.job.Job
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

---

### JobDAG

Directed Acyclic Graph for managing job dependencies.

```{eval-rst}
.. autoclass:: objdet.pipelines.job.JobDAG
   :members:
   :undoc-members:
```

```python
from objdet.pipelines.job import Job, JobDAG, JobType

dag = JobDAG()

# Create jobs with dependencies
preprocess_job = Job(job_type=JobType.PREPROCESS, config={"input_dir": "/data"})
train_job = Job(
    job_type=JobType.TRAIN,
    config={"config_path": "train.yaml"},
    dependencies=[preprocess_job.id],
)

dag.add_job(preprocess_job)
dag.add_job(train_job)

# Get execution order
order = dag.get_execution_order()
```

---

## Celery Tasks

### train_model

```{eval-rst}
.. autofunction:: objdet.pipelines.tasks.train_model
```

---

### export_model

```{eval-rst}
.. autofunction:: objdet.pipelines.tasks.export_model
```

---

### preprocess_data

```{eval-rst}
.. autofunction:: objdet.pipelines.tasks.preprocess_data
```
