# Testing Guide

This guide covers how to run tests for ObjDet, including unit tests and functional tests.

## Test Structure

The test suite is organized into two categories:

```
tests/
├── conftest.py              # Shared fixtures for all tests
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_core/           # Core utilities tests
│   ├── test_data/           # Data loading tests
│   ├── test_models/         # Model tests
│   ├── test_pipelines/      # Pipeline tests
│   └── test_tuning/         # Hyperparameter tuning tests
└── functional/              # Functional tests (end-to-end workflows)
    ├── conftest.py          # Functional test fixtures
    ├── test_cli_training.py # Training workflow tests
    ├── test_cli_inference.py # Inference workflow tests
    └── test_cli_preprocessing.py # Preprocessing tests
```

## Running Tests

### Unit Tests

Unit tests are fast, isolated tests for individual components:

```bash
# Run all unit tests
uv run pytest tests/unit -v

# Run with coverage report
uv run pytest tests/unit --cov=src/objdet --cov-report=term-missing

# Run only fast tests (skip slow ones)
uv run pytest tests/unit -m "not slow"

# Run specific test file
uv run pytest tests/unit/test_data/test_transforms.py -v
```

### Functional Tests

Functional tests verify complete workflows using synthetic sample data:

```bash
# Run all functional tests (excluding slow tests)
uv run pytest tests/functional -v -m "integration and not slow"

# Run training workflow tests only
uv run pytest tests/functional/test_cli_training.py -v

# Run a specific model's training test
uv run pytest tests/functional/test_cli_training.py::TestFasterRCNNTraining -v
uv run pytest tests/functional/test_cli_training.py::TestRetinaNetTraining -v

# Run inference tests
uv run pytest tests/functional/test_cli_inference.py -v

# Run preprocessing tests
uv run pytest tests/functional/test_cli_preprocessing.py -v

# Run slow tests (including LitData conversion)
uv run pytest tests/functional -v -m "slow"
```

## Test Markers

Tests are marked with the following markers:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Tests that take significant time to run |
| `@pytest.mark.integration` | End-to-end functional tests |
| `@pytest.mark.gpu` | Tests that require a GPU |

## Test Fixtures

### Unit Test Fixtures

Common fixtures in `tests/conftest.py`:

- `sample_image` - A 640x640 RGB tensor
- `sample_boxes` - Sample bounding boxes in xyxy format
- `sample_labels` - Sample class labels
- `sample_target` - Complete detection target dict
- `sample_prediction` - Complete prediction dict
- `base_config` - Base configuration dictionary

### Functional Test Fixtures

Fixtures in `tests/functional/conftest.py`:

- `sample_coco_dataset` - Creates a minimal COCO dataset with synthetic images
- `sample_litdata_dataset` - Converts COCO to LitData format
- `faster_rcnn_config` - Faster R-CNN config for testing
- `retinanet_config` - RetinaNet config for testing
- `yolov8_config` - YOLOv8 config for testing
- `litdata_config` - LitData-based config for testing

## Writing Tests

### Unit Test Example

```python
import pytest
import torch

from objdet.models.torchvision import FasterRCNN


class TestFasterRCNN:
    """Tests for Faster R-CNN model."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = FasterRCNN(num_classes=10, pretrained=False)
        assert model is not None

    def test_forward_pass(self, sample_image):
        """Test forward pass produces output."""
        model = FasterRCNN(num_classes=10, pretrained=False)
        model.eval()

        with torch.no_grad():
            output = model([sample_image])

        assert len(output) == 1
        assert "boxes" in output[0]
```

### Functional Test Example

```python
import pytest
from pathlib import Path


@pytest.mark.integration
class TestTrainingWorkflow:
    """Functional tests for training workflows."""

    def test_trainer_fit(self, sample_coco_dataset: Path):
        """Test that training completes successfully."""
        from lightning import Trainer
        from objdet.data.formats.coco import COCODataModule
        from objdet.models.torchvision import FasterRCNN

        model = FasterRCNN(num_classes=2, pretrained=False)
        datamodule = COCODataModule(
            data_dir=sample_coco_dataset,
            batch_size=2,
            num_workers=0,
        )
        trainer = Trainer(fast_dev_run=True, accelerator="cpu")

        trainer.fit(model, datamodule=datamodule)
```

## Web Application Testing

The webapp has comprehensive unit tests for both backend (FastAPI) and frontend (Streamlit) components.

### Running Webapp Tests

**Backend Unit Tests:**

```bash
cd webapp/backend

# Run all unit tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/unit/ --cov=backend --cov-report=term-missing
```

**Coverage:** 51 tests, 82% coverage (100% for API routes and services)

**Frontend Unit Tests:**

```bash
cd webapp/frontend

# Run unit tests
uv run pytest tests/unit/ -v
```

**Coverage:** 15 tests for HTTP client with full endpoint coverage

### Webapp Test Structure

```
webapp/
├── backend/
│   ├── tests/
│   │   ├── conftest.py          # Test fixtures (FastAPI client, mocks)
│   │   ├── unit/
│   │   │   ├── test_celery_service.py    # Celery service tests
│   │   │   ├── test_mlflow_service.py    # MLFlow service tests
│   │   │   ├── test_training_api.py      # Training API tests
│   │   │   ├── test_mlflow_api.py        # MLFlow API tests
│   │   │   └── test_system_api.py        # System API tests
│   │   └── integration/
│   │       └── test_integration.py       # E2E integration tests
│   └── pytest.ini
└── frontend/
    ├── tests/
    │   └── unit/
    │       └── test_client.py    # HTTP client tests with respx
    └── pytest.ini
```

### Integration Tests

Integration tests require running RabbitMQ, MLFlow, and Celery services:

```bash
cd webapp/backend

# Start dependencies with Docker
docker run -d -p 5672:5672 rabbitmq:3
mlflow server --host 0.0.0.0 --port 5000 &
celery -A objdet.pipelines.celery_app worker --loglevel=info &

# Run integration tests
uv run pytest tests/integration/ -v -m integration

# Skip integration tests
uv run pytest -v -m "not integration"
```

## Known Issues

```{warning}
**YOLOv8/YOLOv11 Training Bug**: There is a known issue in the YOLO models
that causes `IndexError: too many indices for tensor of dimension 2` during
training. This affects both CLI and Python API training. The issue is in the
loss computation when processing predictions. TorchVision models (Faster R-CNN,
RetinaNet) are not affected.
```

## Continuous Integration

Tests are automatically run on pull requests via GitHub Actions:

- Unit tests run on every PR
- Functional tests run on main branch merges
- GPU tests run on scheduled nightly builds (if GPU runners available)

## Troubleshooting

### Common Issues

**Tests are slow**: Use the `-m "not slow"` marker to skip slow tests during development.

**CUDA out of memory**: The test cleanup fixture should clear CUDA memory after each test.
If issues persist, try running tests with `--forked` to isolate CUDA state.

**Import errors**: Ensure you've installed the package with `uv sync --all-extras`.
