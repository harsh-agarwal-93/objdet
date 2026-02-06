"""Unit tests for Celery tasks."""

from __future__ import annotations


def test_train_model_task_registered() -> None:
    """Test train_model task is registered with Celery."""
    # Import tasks to trigger registration
    import backend.tasks  # noqa: F401
    from backend.celery_app import app

    # Verify task is registered
    # Verify task is registered
    assert "backend.tasks.train_model" in app.tasks
    assert callable(app.tasks["backend.tasks.train_model"])


def test_train_model_execution() -> None:
    """Test train_model task execution logic."""
    from unittest.mock import patch

    from backend.tasks import train_model

    # Mock the update_state method on the task instance
    # properly mocking bound task methods is tricky, so we use execute/apply
    # and mock the request context or reliance on state

    # Simple execution via apply()
    # We patch time.sleep to speed up test
    with patch("time.sleep"):
        result = train_model.apply(  # type: ignore[not-callable]
            kwargs={
                "config_path": "config.yaml",
                "output_dir": "output",
                "max_epochs": 10,
            }
        ).result

    assert result["status"] == "success"
    assert result["epochs_completed"] == 10
    assert "final_metrics" in result


def test_export_model_execution() -> None:
    """Test export_model task execution logic."""
    from unittest.mock import patch

    from backend.tasks import export_model

    with patch("time.sleep"):
        result = export_model.apply(  # type: ignore[not-callable]
            kwargs={"checkpoint_path": "model.pt", "export_format": "onnx"}
        ).result

    assert result["status"] == "success"
    assert result["exported_path"] == "model.pt.onnx"


def test_preprocess_data_execution() -> None:
    """Test preprocess_data task execution logic."""
    from unittest.mock import patch

    from backend.tasks import preprocess_data

    with patch("time.sleep"):
        result = preprocess_data.apply(  # type: ignore[not-callable]
            kwargs={"input_dir": "input", "output_dir": "output", "format_name": "coco"}
        ).result

    assert result["status"] == "success"
    assert result["num_samples_processed"] == 1000
    assert result["num_samples_processed"] == 1000


def test_export_model_task_registered() -> None:
    """Test export_model task is registered with Celery."""
    import backend.tasks  # noqa: F401
    from backend.celery_app import app

    # Verify task is registered
    assert "backend.tasks.export_model" in app.tasks
    assert callable(app.tasks["backend.tasks.export_model"])


def test_preprocess_data_task_registered() -> None:
    """Test preprocess_data task is registered with Celery."""
    import backend.tasks  # noqa: F401
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
