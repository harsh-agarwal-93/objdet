# Backend Service

This directory contains the FastAPI backend service for the ObjDet web application.

## Structure
- `app/`: FastAPI application code.
- `celery_app.py`: Celery worker configuration.
- `main.py`: Application entry point.

## Development
Run locally:
```bash
uv run uvicorn backend.main:app --reload
```
