# ObjDet Web Application

A  modern web application for managing object detection model training, MLFlow experiments, and deployment pipelines.

## Architecture

```
┌─────────────────────┐
│  Streamlit Frontend │  (Port 8501)
│   (User Interface)  │
└──────────┬──────────┘
           │ HTTP/REST
           ▼
┌─────────────────────┐
│   FastAPI Backend   │  (Port 8000)
│   (REST API Layer)  │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌─────────┐
│  Celery │  │ MLFlow  │
│  Tasks  │  │ Tracking│
└─────────┘  └─────────┘
```

## Features

### ✅ Implemented

**Backend (FastAPI):**
- REST API with automatic OpenAPI docs
- Celery task submission and monitoring
- MLFlow data retrieval (experiments, runs, metrics, artifacts)
- System health monitoring
- CORS support for Streamlit frontend

**Frontend (Streamlit):**
- **Models Page (Workflow 1):** Full training job submission, MLFlow run browsing, active job monitoring
- **Home Dashboard:** System status, quick navigation
- **Placeholder Pages:** Effects, Synthetic Data, SceneForge, Loadset with planned UI structure

**Testing:**
- **Backend:** 51 unit tests with 82% code coverage
- **Frontend:** 15 unit tests for HTTP client
- Integration test structure in place

### 🚧 Coming Soon

- Effects training (Workflow 2)
- Synthetic data generation (Workflow 3)
- WebSocket support for real-time updates

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose (for containerized deployment)

### Option 1: Docker Compose (Recommended)

```bash
cd webapp
docker-compose up
```

This starts all services:
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/api/docs
- **MLFlow UI:** http://localhost:5000
- **RabbitMQ Management:** http://localhost:15672 (guest/guest)

### Option 2: Local Development

**1. Install Backend Dependencies:**

```bash
cd webapp/backend
uv sync
```

**2. Install Frontend Dependencies:**

```bash
cd webapp/frontend
uv sync
```

**3. Start Services:**

```bash
# Terminal 1: Start RabbitMQ (required for Celery)
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# Terminal 2: Start MLFlow server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 3: Start Celery worker (from repository root)
cd ../../  # back to repo root
celery -A objdet.pipelines.celery_app worker --loglevel=info

# Terminal 4: Start FastAPI backend
cd webapp/backend
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 5: Start Streamlit frontend
cd webapp/frontend
uv run streamlit run app.py
```

**4. Access the Application:**

- Frontend: http://localhost:8501
- Backend API Docs: http://localhost:8000/api/docs

## Environment Variables

### Backend

Create a `.env` file in `webapp/backend/`:

```bash
# Celery Configuration
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=rpc://

# MLFlow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=objdet

# API Configuration
API_TITLE=ObjDet WebApp API
API_VERSION=0.1.0
```

### Frontend

Create a `.env` file in `webapp/frontend/`:

```bash
# Backend URL
BACKEND_URL=http://localhost:8000
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc

### Key Endpoints

**Training:**
- `POST /api/training/submit` - Submit training job
- `GET /api/training/status/{task_id}` - Get task status
- `POST /api/training/cancel/{task_id}` - Cancel task
- `GET /api/training/active` - List active tasks

**MLFlow:**
- `GET /api/mlflow/experiments` - List experiments
- `GET /api/mlflow/runs` - List runs
- `GET /api/mlflow/runs/{run_id}` - Get run details
- `GET /api/mlflow/runs/{run_id}/metrics` - Get metrics
- `GET /api/mlflow/runs/{run_id}/artifacts` - List artifacts

**System:**
- `GET /health` - Health check
- `GET /api/system/status` - Service status
- `GET /api/system/config` - System configuration

## Project Structure

```
webapp/
├── backend/                  # FastAPI backend
│   ├── api/                 # API routes
│   │   ├── training.py      # Training endpoints
│   │   ├── mlflow.py        # MLFlow endpoints
│   │   └── system.py        # System endpoints
│   ├── services/            # Business logic
│   │   ├── celery_service.py
│   │   └── mlflow_service.py
│   ├── models/              # Pydantic models
│   │   ├── training.py
│   │   └── mlflow.py
│   ├── core/                # Configuration
│   │   └── config.py
│   ├── main.py              # FastAPI app
│   └── Dockerfile
├── frontend/                 # Streamlit frontend
│   ├── pages/               # Streamlit pages
│   │   ├── 2_🧠_Models.py
│   │   ├── 3_⚡_Effects.py
│   │   ├── 4_📦_Synthetic_Data.py
│   │   ├── 5_🎬_SceneForge.py
│   │   └── 6_📁_Loadset.py
│   ├── components/          # Reusable UI components
│   │   ├── status_badge.py
│   │   ├── metrics_chart.py
│   │   └── job_monitor.py
│   ├── api/                 # Backend HTTP client
│   │   └── client.py
│   ├── utils/               # Utilities
│   │   ├── formatting.py
│   │   └── session.py
│   ├── app.py               # Main Streamlit app
│   └── Dockerfile
└── docker-compose.yml        # Multi-container deployment
```

## Usage Guide

### Training a Model

1. Navigate to the **Models** page in the Streamlit UI
2. Click the **New Training** tab
3. Fill out the training configuration:
   - Run name, model architecture, dataset
   - Hyperparameters (epochs, batch size, learning rate)
   - GPU selection and priority
4. Click **Start Training**
5. Monitor progress in the **Active Jobs** tab
6. View completed runs in the **Previous Runs** tab

### Viewing Training Metrics

1. Go to the **Models** page
2. In the **Previous Runs** tab, click on a run
3. Click **View Details** to see:
   - Training parameters
   - Final metrics
   - Live training curves (loss, accuracy)

## Development

### Adding New API Endpoints

1. Define Pydantic models in `backend/models/`
2. Implement service logic in `backend/services/`
3. Create API router in `backend/api/`
4. Register router in `backend/main.py`

### Adding New UI Pages

1. Create page file in `frontend/pages/` (e.g., `7_📊_Analytics.py`)
2. Use components from `frontend/components/`
3. Call backend via `frontend/api/client.py`
4. Update navigation in `frontend/app.py`

## Testing

The webapp has comprehensive unit tests for both backend and frontend components.

### Backend Unit Tests

**Run all unit tests:**

```bash
cd webapp/backend
uv run pytest tests/unit/ -v
```

**Run with coverage:**

```bash
cd webapp/backend
uv run pytest tests/unit/ -v --cov=backend --cov-report=term-missing
```

**Current test coverage:**
- **51 unit tests** covering service layer and API endpoints
- **82% code coverage** with 100% coverage for API routes and services
- Tests for: Celery task management, MLFlow integration, all API endpoints, error handling

**Integration Tests** (requires running RabbitMQ, MLFlow, Celery):

```bash
cd webapp/backend
# Start dependencies first
docker run -d -p 5672:5672 rabbitmq:3
mlflow server --host 0.0.0.0 --port 5000 &
celery -A objdet.pipelines.celery_app worker --loglevel=info &

# Run integration tests
uv run pytest tests/integration/ -v -m integration
```

### Frontend Unit Tests

**Run all frontend tests:**

```bash
cd webapp/frontend
uv run pytest tests/unit/ -v
```

**Current test coverage:**
- **15 unit tests** for HTTP client (`BackendClient`)
- Tests for: All API endpoints, error handling, HTTP mocking, singleton pattern
- Uses `respx` library for HTTP mocking (no actual backend required)

### Continuous Integration

Tests are marked with pytest markers for selective execution:
- Skip integration tests: `pytest -v -m "not integration"`
- Run only integration tests: `pytest -v -m integration`

## Deployment

### Production Considerations

- Use environment-specific `.env` files
- Configure reverse proxy (nginx/Traefik) for HTTPS
- Set up persistent volumes for MLFlow artifacts
- Scale Celery workers based on load
- Enable authentication (OAuth, JWT)

### Cloud Deployment

**Google Cloud Run:**
- Build backend/frontend containers separately
- Deploy with Cloud Build
- Use Cloud SQL for MLFlow backend store

**AWS ECS/Fargate:**
- Use Task Definitions for each service
- Configure ALB for load balancing
- Store artifacts in S3

**Kubernetes:**
- Create Helm chart with backend, frontend, RabbitMQ, MLFlow
- Use PersistentVolumeClaims for MLFlow data
- Configure Ingress for routing

## Troubleshooting

**"Failed to connect to backend"**
- Ensure backend is running on port 8000
- Check `BACKEND_URL` environment variable
- Verify CORS settings in `backend/core/config.py`

**"Celery task submission failed"**
- Verify RabbitMQ is running and accessible
- Check Celery worker logs
- Ensure `CELERY_BROKER_URL` is correct

**"No training runs found"**
- Verify MLFlow server is running
- Check `MLFLOW_TRACKING_URI` configuration
- Ensure at least one training job has been submitted

## Contributing

Please ensure all contributions include:
- Type hints for Python code
- Docstrings for functions/classes
- Unit tests for new features
- Updated documentation

## License

See repository root for license information.
