# Deployment Guide

This guide covers how to build, deploy, and serve models using the ObjDet framework.

## Docker Infrastructure

The project uses a consolidated multi-stage Dockerfile at `deploy/docker/ml.Dockerfile` for both training and serving.

### Building Images

You must run build commands from the project root context.

#### Training Image
Contains the full training environment, including Celery worker dependencies.

```bash
docker build \
    -f deploy/docker/ml.Dockerfile \
    --target train \
    -t objdet:train .
```

#### Serving Image
Optimized for inference with LitServe.

```bash
docker build \
    -f deploy/docker/ml.Dockerfile \
    --target serve \
    -t objdet:serve .
```

### Running with Docker Compose

The `deploy/docker-compose.yml` file orchestrates the entire stack:

- **RabbitMQ**: Message broker for job queues
- **MLflow**: Experiment tracking server (v3.9.0)
- **ML Worker**: Celery worker for processing training jobs
- **Serve**: Inference API
- **Backend**: FastAPI web backend
- **Frontend**: React web UI

To start the stack:

```bash
docker-compose -f deploy/docker-compose.yml up -d
```

To view logs:

```bash
docker-compose -f deploy/docker-compose.yml logs -f
```

## Model Serving

### Local Serving
You can serve a model locally using the CLI:

```bash
objdet serve --config configs/serving/default.yaml
```

The server exposes:
- `POST /predict`: Inference endpoint
- `GET /health`: Health check

### Production Serving

For production, use the `objdet:serve` Docker image. Ensure you mount your model checkpoints and configurations:

```yaml
services:
  serve:
    image: objdet:serve
    volumes:
      - ./models:/app/models:ro
      - ./configs:/app/ml/configs:ro
    environment:
      - MODEL_PATH=/app/models/best.ckpt
    ports:
      - "8000:8000"
```

## Model Export

Before deployment, you may want to export your model to an optimized format like ONNX or TensorRT.

### Export to ONNX

```bash
objdet export \
    --checkpoint checkpoints/best.ckpt \
    --format onnx \
    --output models/model.onnx \
    --input_shape 1 3 640 640
```

### Export to TensorRT

TensorRT export requires a GPU environment:

```bash
objdet export \
    --checkpoint checkpoints/best.ckpt \
    --format tensorrt \
    --output models/model.plan
```

## Health Checks

All services implement health checks:

- **MLflow**: `http://localhost:5000/health`
- **Serve**: `http://localhost:8000/health`
- **Backend**: `http://localhost:8000/health`
