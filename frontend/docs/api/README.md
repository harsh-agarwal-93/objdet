# ObjDet API Collection

This directory contains API documentation and testing resources for the ObjDet backend.

## Hoppscotch Collection

[Hoppscotch](https://hoppscotch.io/) is an open-source API development ecosystem for testing REST APIs.

### Import Instructions

1. Open [Hoppscotch](https://hoppscotch.io/) in your browser
2. Click **Collections** in the left sidebar
3. Click the **Import** button (↓ icon)
4. Select **Import from File**
5. Choose `hoppscotch-collection.json` from this directory

### Environments

The collection includes two pre-configured environments:

| Environment | Backend URL | Use Case |
|-------------|-------------|----------|
| **Local Development** | `http://localhost:8000` | Local dev server |
| **Docker Compose** | `http://backend:8000` | Docker network |

### Available Endpoints

#### Training
- `POST /api/training/submit` - Submit a new training job
- `GET /api/training/status/{task_id}` - Get task status
- `POST /api/training/cancel/{task_id}` - Cancel a running task
- `GET /api/training/active` - List all active tasks

#### MLFlow
- `GET /api/mlflow/experiments` - List experiments
- `GET /api/mlflow/runs` - List runs (with filters)
- `GET /api/mlflow/runs/{run_id}` - Get run details
- `GET /api/mlflow/runs/{run_id}/metrics` - Get run metrics
- `GET /api/mlflow/runs/{run_id}/artifacts` - List artifacts

#### System
- `GET /api/system/status` - Check system health

### Variables

Set these variables in your Hoppscotch environment:

| Variable | Description | Example |
|----------|-------------|---------|
| `BACKEND_URL` | Backend API base URL | `http://localhost:8000` |
| `TASK_ID` | Current task ID (set after submit) | `abc123...` |
| `RUN_ID` | MLFlow run ID | `run-xyz...` |

## OpenAPI Specification

See `openapi.yaml` for the complete OpenAPI 3.0 specification of all endpoints.
