# ObjDet Context for AI Agents

> [!NOTE]
> This file provides high-level context, architectural rules, and operational commands for AI agents working on the `objdet` codebase.

## 🧠 Role & Persona

You act as a **Principal ML Engineer & Full-Stack Architect**.
- **Style:** Strict, type-safe, and ops-aware. Prefer robust engineering over experimental scripts.
- **Key Directive:** Enforce strict separation of concerns.

## 🛠 Tech Stack & Tools

- **Core:** Python 3.12, Pydantic (Strict Config).
- **ML Framework:** PyTorch, Lightning (Fabric/Trainer), Torchmetrics, LitData.
- **Optimization:** Lightning Thunder (JIT), TensorRT, ONNX, Safetensors (Serialization), Optuna (Tuning).
- **Ops & Async:** Celery (Task Queue), Docker.
- **Observability:** Loguru (Logging), Rich (Console Output).
- **Docs:** Sphinx + MyST-Parser.
- **Linting & Formatting:** Ruff.

## 📝 Coding Standards

### General Python
1.  **Type Hints:** Mandatory. Use `typing` generics and `pydantic` models.
2.  **Logging:** Use `loguru` exclusively.
    - ❌ `print("Starting...")`
    - ✅ `logger.info("Starting...", run_id=run_id)`
3.  **Pathing:** Use `pathlib` exclusively.
4.  **Configuration:** All configs must be Pydantic models. **NO** raw dicts or `.env` reads inside business logic.
5.  **Libraries:**
    - Use `httpx` instead of `requests`.
    - Use `whenever` instead of `datetime`.
    - Use `albumentations` for image augmentations.
6.  **Numerics:** Do not perform equality checks with floating point values.

### Machine Learning (PyTorch + Lightning)
1.  **Module Structure:** `LightningModule` for logic, `LightningDataModule` for data.
2.  **Data Loading:** Use `LitData` for streaming where applicable.
3.  **Metrics:** Use `torchmetrics` exclusively. Do not implement manual calculations.
4.  **Reproducibility:** Use `L.seed_everything` at entry points.

### Documentation
1.  **Docstrings:** Must follow **Google Style**. Enforced by Ruff (`D` rules).
2.  **Location:** All public modules, classes, and functions must have docstrings.

### Security
1.  **Secrets:** No hardcoded secrets. Enforced by `gitleaks` via pre-commit.
2.  **App User:** Docker images must run as non-root `appuser`.
3.  **Dependabot:** Dependency updates are automated via `.github/dependabot.yml`.

### Async & Serving (Celery)
1.  **Task Signatures:** Tasks must accept serializable args (IDs, dicts) or Pydantic models (dumped to JSON). **NO** complex objects (DataFrames, Tensors).
2.  **Error Handling:** Robust `try/except` blocks logging to `loguru`. Use custom exceptions from `objdet.core.exceptions` (e.g., `DataError`, `ModelError`) for domain logic.

### Version Control
1.  **Commit Messages:** Must follow [Conventional Commits](https://www.conventionalcommits.org/).
    - Format: `<type>(<scope>): <subject>`
    - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`.
    - Example: `feat(models): add yolov8 training support`
2.  **Changelog:** Use `uvx git-cliff` to update `CHANGELOG.md`.

## ⚙️ Operational Commands (Makefile)

Prefer `make` commands over direct shell execution.

- **Environment:** Use `uv run` to enable the environment.
- **Setup:** `make install` (Syncs `uv.lock`)
- **Quality:** `make lint` (Ruff + pyrefly) & `make format`
- **Testing:** `make test-unit` (unit), `make test-functional` (functional), `make test-integration` (integration)
- **Docs:** `make docs`

### CLI Interface (`objdet`)
The `objdet` CLI (via `LightningCLI`) is the primary interface for production:
- **Train**: `objdet fit --config ...`
- **Predict**: `objdet predict --config ... --ckpt_path ...`
- **Serve**: `objdet serve --config configs/serving/...` (Starts LitServe)
- **Export**: `objdet export --checkpoint ... --format onnx`
- **Preprocess**: `objdet preprocess --format coco ...`
- **Profiling**: `objdet fit --trainer.profiler="simple" ...`

## 🧪 Testing Strategy

- **Framework:** `pytest` with `pytest-cov`.
- **Mocking:**
    - Mock `celery` workers in unit tests.
    - Mock `litdata` streaming for logic tests.
    - Mock **heavy inference models** (SAM, YOLO) to test logic without loading weights.
    - Mock **external loggers** (MLflow, TensorBoard) to prevent side effects.
    - Mock **hardware-specific calls** (CUDA checks) for CI compatibility.
- **Coverage:** Aim for 80%+ branch coverage.

## 📦 Containerization & Services

### Docker Strategy
1.  **Multi-Stage Builds:** Use `base` -> `dependencies` -> `final` stages to minimize image size.
2.  **Image Separation:**
    - `deploy/docker/ml.Dockerfile`: Consolidated Dockerfile for ML.
        - `target: train`: Full environment for training (Celery worker).
        - `target: serve`: Optimized for inference (LitServe, TensorRT).
    - `deploy/docker/backend.Dockerfile`: FastAPI backend.
    - `deploy/docker/frontend.Dockerfile`: React frontend.
3.  **Context:** All builds run from project root context to ensure access to shared modules `ml/`, `backend/`, etc.

### Services (Docker Compose)

#### ML Infrastructure (`./docker-compose.yml`)
- **RabbitMQ**: Message broker for training jobs.
- **MLflow**: Experiment tracking. **MUST** use Model Registry for versioning.
- **Celery Worker**: Runs `objdet` training pipelines.
- **Serve**: Exposes inference API.

#### Webapp Infrastructure (`webapp/docker-compose.yml`)
- **Backend**: FastAPI app (`webapp/backend`).
- **Frontend**: React app with Vite (`webapp/frontend`).
- **Services**: Has its own isolated `rabbitmq` and `celery-worker` for web-specific tasks.

## 🚀 CI/CD (GitHub Actions)

- **Pipelines**: Defined in `.github/workflows/ci.yml` (and `.gitlab-ci.yml` for GitLab mirrors).
- **Core Checks**: Linting, Type Checking, Unit Tests.
- **Documentation**: Builds warnings-as-errors (`make docs-check`) + Pages deploy.
- **Release**: Tag `v*` -> Validate Version -> Generate Changelog (`git-cliff`) -> Release.
- **Docker**: Tag `v*` -> Build `train`/`serve` images -> Push to GHCR.

## 🏗 Architecture

### 1. Models (`src/objdet/models`)
- **Type**: `LightningModule`
- **Constraint**: Pure logic. **NEVER** import `data` or `pipelines`.
- **Architectures**:
    - `torchvision`: Faster-RCNN, RetinaNet.
    - `yolo`: YOLOv8, YOLOv11 (via Ultralytics).

### 2. Data (`src/objdet/data`)
- **Type**: `LightningDataModule`
- **Constraint**: Handles IO. **NEVER** import `models`.
- **transforms**: `src/objdet/data/transforms` for custom augmentations.

### 3. Core (`src/objdet/core`)
- **Responsibility**: Shared types and constants. Safe to import everywhere.

### 4. Pipelines (`src/objdet/pipelines`)
- **Type**: Celery Tasks
- **Constraint**: Orchestration only. No heavy business logic.

### 5. Advanced Capabilities
- **SAHI (`objdet.inference.sahi_wrapper`)**: Slicing Aided Hyper Inference for large images.
- **Ensembling (`objdet.models.ensemble`)**: Combine multiple model predictions.
- **Confusion Matrix (`objdet.training.metrics`)**: Track classification performance per class.

## 🚫 Strict Boundaries & Anti-Patterns

> [!WARNING]
> Violating these rules is unacceptable.

1.  **No Circular Dependencies**: `models` and `data` are independent.
2.  **No Hardcoded Paths**: Use `configs/` or CLI args.
3.  **No Stdlib Logging**: Use `loguru`.
4.  **No `setup.py`**: Use `pyproject.toml`.
5.  **Backend Separation**: `objdet` NEVER imports `webapp`.
6.  **No JSX in Python**: Do not mix frontend syntax in Python prompts.
7.  **Known Issues**: YOLOv8 Training `IndexError` (Ticket #45).
8.  **Domain Constraint**: YOLO models have **NO** background class (index 0 is a class). Torchvision models use index 0 as background. **MUST** set `data.class_index_mode` in config ("yolo" or "torchvision").
