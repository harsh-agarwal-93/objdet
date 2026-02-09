# Contributing to ObjDet

Thank you for your interest in contributing to ObjDet! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Adding New Models](#adding-new-models)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something together.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/harsh-agarwal-93/objdet.git`
3. Add the upstream remote: `git remote add upstream https://github.com/originalowner/objdet.git`

## Development Setup

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies including dev tools
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Configure git commit template
git config commit.template .gitmessage
```

## Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```

2. Make your changes following our [code style](#code-style)

3. Write tests for new functionality

4. Run the test suite:
   ```bash
   uv run pytest tests/unit -v
   ```

5. Run linting and type checking:
   ```bash
   uv run ruff check src tests
   uv run ruff format src tests
   uv run pyrefly check src
   ```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages. This enables automatic changelog generation.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring |
| `perf` | Performance improvements |
| `test` | Adding or updating tests |
| `build` | Build system changes |
| `ci` | CI/CD changes |
| `chore` | Other changes |
| `revert` | Reverting a previous commit |

### Scopes

Common scopes include: `models`, `data`, `training`, `inference`, `serving`, `pipelines`, `cli`

### Examples

```
feat(models): add YOLOv11 lightning module

Implements YOLOv11 as a PyTorch Lightning module, wrapping the
Ultralytics model architecture with custom training/validation steps.

Closes #123
```

## Pull Request Process

1. Ensure your branch is up to date with `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your branch and create a pull request

3. Fill out the PR template completely

4. Ensure all CI checks pass

5. Request review from maintainers

6. Address review feedback

7. Once approved, your PR will be merged

## Adding New Models

To add a new object detection model:

1. Create a new file in `src/objdet/models/` (or appropriate subdirectory)

2. Inherit from `BaseLightningDetector`:
   ```python
   from objdet.models.base import BaseLightningDetector

   class MyNewModel(BaseLightningDetector):
       """My new object detection model.

       Args:
           num_classes: Number of classes to detect.
           pretrained: Whether to use pretrained weights.
       """

       def __init__(self, num_classes: int, pretrained: bool = True) -> None:
           super().__init__()
           # Initialize your model
   ```

3. Register the model in `src/objdet/models/registry.py`

4. Create a config file in `configs/model/`

5. Write unit tests in `tests/unit/test_models/`

6. Update documentation

## Code Style

- **Line length**: 100 characters
- **Docstrings**: Google style, required for all public classes and functions
- **Type hints**: Required for all function signatures
- **Imports**: Sorted with `ruff` (isort rules)

### Example

```python
"""Module docstring describing the module's purpose."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence


class MyClass:
    """Class docstring with description.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Attributes:
        attr1: Description of attr1.
    """

    def __init__(self, param1: int, param2: str = "default") -> None:
        self.attr1 = param1
        self._param2 = param2

    def my_method(self, x: Tensor) -> Tensor:
        """Method docstring.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output tensor of shape (N, num_classes).

        Raises:
            ValueError: If input has wrong dimensions.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D")
        return x
```

## Testing

We use pytest for testing with two categories of tests:

- **Unit tests** (`tests/unit/`): Fast, isolated tests for individual components
- **Functional tests** (`tests/functional/`): End-to-end workflow tests with sample data

### Running Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit -v

# Run with coverage
uv run pytest tests/unit --cov=src/objdet --cov-report=term-missing

# Run only fast tests
uv run pytest tests/unit -m "not slow"
```

### Running Functional Tests

Functional tests verify complete workflows (training, inference, preprocessing) using synthetic sample data:

```bash
# Run all functional tests (excluding slow tests)
uv run pytest tests/functional -v -m "integration and not slow"

# Run training workflow tests
uv run pytest tests/functional/test_cli_training.py -v

# Run specific model training test
uv run pytest tests/functional/test_cli_training.py::TestFasterRCNNTraining -v

# Run inference tests
uv run pytest tests/functional/test_cli_inference.py -v

# Run preprocessing tests
uv run pytest tests/functional/test_cli_preprocessing.py -v
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Tests that take significant time |
| `@pytest.mark.integration` | End-to-end functional tests |
| `@pytest.mark.gpu` | Tests requiring GPU |

### Writing Tests

- Write tests for all new functionality
- Aim for 80% code coverage
- Use pytest fixtures for common setup
- Use the `sample_coco_dataset` fixture for functional tests with sample data

### Known Issues

> [!WARNING]
> **YOLOv8 Training**: There is a known bug in the YOLOv8 model that causes
> `IndexError: too many indices for tensor of dimension 2` during training.
> This affects both CLI and Python API training. The issue is in the loss
> computation when processing predictions. YOLOv11 may have the same issue.
> See `tests/functional/test_cli_training.py::TestYOLOv8Training` for details.

## Documentation

- Update docstrings for any changed APIs
- Add docstrings to new classes and functions
- Update README if adding user-facing features
- Add entries to CHANGELOG.md under `[Unreleased]`
- For significant changes, consider adding to `docs/`

### Building Documentation

```bash
# Build docs locally
uv run sphinx-build -b html docs docs/_build/html

# View in browser
open docs/_build/html/index.html
```

## Questions?

If you have questions, please open an issue with the `question` label.

Thank you for contributing! 🎉
