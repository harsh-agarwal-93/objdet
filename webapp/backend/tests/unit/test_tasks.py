"""Unit tests for Celery tasks."""

from __future__ import annotations


def test_train_model_task_registered() -> None:
    """Test train_model task is registered with Celery."""
    from backend.celery_app import app

    # Verify task is registered
    assert "backend.tasks.train_model" in app.tasks
    assert callable(app.tasks["backend.tasks.train_model"])


def test_export_model_task_registered() -> None:
    """Test export_model task is registered with Celery."""
    from backend.celery_app import app

    # Verify task is registered
    assert "backend.tasks.export_model" in app.tasks
    assert callable(app.tasks["backend.tasks.export_model"])


def test_preprocess_data_task_registered() -> None:
    """Test preprocess_data task is registered with Celery."""
    from backend.celery_app import app

    # Verify task is registered
    assert "backend.tasks.preprocess_data" in app.tasks
    assert callable(app.tasks["backend.tasks.preprocess_data"])


def test_train_model_task_signature() -> None:
    """Test train_model task signature and parameters."""
    from backend.tasks import train_model

    # Verify task properties
    assert train_model.name == "backend.tasks.train_model"
    # Task is bound, so it has 'request' as first parameter
    assert hasattr(train_model, "request")


def test_task_metadata() -> None:
    """Test task configuration and metadata."""
    from backend.celery_app import app

    task = app.tasks["backend.tasks.train_model"]

    # Verify task name
    assert task.name == "backend.tasks.train_model"

    # Task should be callable
    assert callable(task)
