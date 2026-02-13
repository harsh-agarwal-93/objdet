# Deployment Configuration

This directory contains Docker and Docker Compose configurations for deploying the ObjDet system.

## Structure

```
deploy/
├── docker-compose.yml          # Unified composition for all services
├── docker/
│   ├── ml.Dockerfile           # ML image (multi-target: train, serve)
│   ├── backend.Dockerfile      # FastAPI backend
│   └── frontend.Dockerfile     # React frontend (Nginx)
└── README.md
```

## Services

| Service | Description | Port |
|---------|-------------|------|
| `rabbitmq` | Message broker (AMQP + Management UI) | 5672, 15672 |
| `mlflow` | Experiment tracking | 5000 |
| `ml-worker` | Celery worker for ML training | — |
| `serve` | LitServe inference API | 8001 |
| `backend` | FastAPI web API | 8000 |
| `webapp-worker` | Celery worker for web tasks | — |
| `frontend` | React SPA via Nginx | 3000 |
| `sonarqube` | Code quality analysis | 9000 |

## Usage

Build and start all services:
```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

Build individual ML targets:
```bash
# Training image
docker build --target train -f deploy/docker/ml.Dockerfile -t objdet-train .

# Serving image
docker build --target serve -f deploy/docker/ml.Dockerfile -t objdet-serve .
```
