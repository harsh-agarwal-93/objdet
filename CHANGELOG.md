# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🧪 Testing

- **Unit Tests**: Added 62+ unit tests covering core modules:
    - `test_logging.py`: Comprehensive tests for logging utilities (JSON formatting, file handlers, log levels).
    - `test_base.py`: Tests for base data module functionality.
    - `test_voc.py`: Tests for Pascal VOC dataset parsing and loading.
    - `test_yolo.py`: Tests for YOLO format dataset handling.
    - `test_ensemble.py`: Tests for model ensemble strategies (WBF, NMS).
- **Integration Tests**: Added full integration test suite:
    - `test_cli.py`: End-to-end CLI command tests (fit, predict, validate, export, serve, preprocess).
    - `test_serving.py`: LitServe API integration tests with A/B testing support.
    - `test_callbacks.py`: Training callback integration tests (checkpointing, early stopping, logging).
    - `test_export.py`: Model export tests for ONNX, TensorRT, and SafeTensors formats.
- **Test Infrastructure**: Added `conftest.py` with shared fixtures for integration tests.

### 🚀 Features

- **Models**: Added support for Faster R-CNN, RetinaNet, YOLOv8, and YOLOv11 architectures.
- **Configs**: Added full experiment configs combining model, data, and trainer sections:
    - COCO configs: `faster_rcnn_coco.yaml`, `retinanet_coco.yaml`, `yolov8_coco.yaml`, `yolov11_coco.yaml`
    - LitData configs: `faster_rcnn_litdata.yaml`, `retinanet_litdata.yaml`, `yolov8_litdata.yaml`, `yolov11_litdata.yaml`
- **MLOps**:
    - **Serving**: Implemented LitServe for high-performance model serving with dynamic batching and A/B testing.
    - **Pipelines**: Integrated RabbitMQ and Celery for distributed job queuing and management.
    - **Data**: Added LitData support for optimized dataset loading and streaming.
    - **Optimization**: Added Optuna for automated hyperparameter tuning.
- **Inference**: Added SAHI integration for large image sliced inference.
- **Exceptions**: Added `DependencyError` for clear messaging when optional dependencies are missing.

### 🚜 Refactor

- **Dependencies**:
    - Replaced `requests` with `httpx` for modern, async-capable HTTP clients.
    - Replaced `datetime` with `whenever` for robust time handling.
- **Typing**: Migrated from Pyright to **Pyrefly** for stricter, more accurate type checking.
- **Data**: Made `LitDataConverter.input_dir` optional when format-specific paths are provided directly.
- **Data**: Refactored `LitDataDataModule` to use native `StreamingDataset` with `transform` parameter and `StreamingDataLoader` with custom `collate_fn`. Added `DetectionStreamingDataset` factory and `create_streaming_dataloader` helper for full LitData API compatibility.

### 🛠 Maintenance

- **CI/CD**:
    - Updated `pre-commit` hooks to use `uv` and `pyrefly`.
    - Enforced strict `ruff` linting and formatting rules.
    - Enhanced `dependabot.yml` for comprehensive `uv` and Docker dependency tracking.
- **Build System**: Fully migrated to `uv` for package management and project synchronization.

### 📚 Documentation

- Added comprehensive README with project overview, installation, and usage examples.
- Added preprocessing documentation with CLI and Python API examples.
- Added comprehensive data formats guide with LitData streaming, COCO, VOC, and YOLO format details.
- Added data API reference with full documentation for all data modules and utilities.
- **API Documentation**: Updated all API docs with Sphinx/MyST autodoc syntax:
    - `inference.md`: Documented `Predictor` and `SlicedInference` (SAHI).
    - `models.md`: Documented model registry, base class, and all model implementations.
    - `pipelines.md`: Documented SDK functions, Job models, and Celery tasks.
    - `serving.md`: Documented `DetectionAPI` and `ABTestingAPI` with examples.
    - `training.md`: Documented callbacks and metrics.

### 🛠 Build Automation

- **Makefile**: Added convenience scripts for common development tasks:
    - `make docs` / `docs-serve` / `docs-check` / `docs-linkcheck` for documentation.
    - `make lint` / `format` / `pre-commit` for code quality.
    - `make test` / `test-unit` / `test-cov` for testing.
    - `make clean` for artifact cleanup.
- **CI**: Updated docs job to use `make docs-check` for stricter validation (warnings as errors).
