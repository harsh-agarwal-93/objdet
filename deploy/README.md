# Deployment Configuration

This directory contains Docker and Docker Compose configurations for deploying the ObjDet system.

## Files
- `docker-compose.yml`: Unified composition for all services (ML, Backend, Frontend, RabbitMQ, MLflow).
- `Dockerfile.train`: Docker image for ML training worker.
- `Dockerfile.serve`: Docker image for ML inference server.

## Usage
Build and start all services:
```bash
docker compose -f deploy/docker-compose.yml up --build -d
```
