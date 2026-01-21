# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Features

- **Models**: Added support for Faster R-CNN, RetinaNet, YOLOv8, and YOLOv11 architectures.
- **MLOps**:
    - **Serving**: Implemented LitServe for high-performance model serving with dynamic batching and A/B testing.
    - **Pipelines**: Integrated RabbitMQ and Celery for distributed job queuing and management.
    - **Data**: Added LitData support for optimized dataset loading and streaming.
    - **Optimization**: Added Optuna for automated hyperparameter tuning.
- **Inference**: Added SAHI integration for large image sliced inference.

### 🚜 Refactor

- **Dependencies**:
    - Replaced `requests` with `httpx` for modern, async-capable HTTP clients.
    - Replaced `datetime` with `whenever` for robust time handling.
- **Typing**: Migrated from Pyright to **Pyrefly** for stricter, more accurate type checking.

### 🛠 Maintenance

- **CI/CD**:
    - Updated `pre-commit` hooks to use `uv` and `pyrefly`.
    - Enforced strict `ruff` linting and formatting rules.
- **Build System**: Fully migrated to `uv` for package management and project synchronization.

### 📚 Documentation

- Add README with project overview
