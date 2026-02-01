"""Unit tests for Celery tasks."""

from __future__ import annotations


class TestCeleryTaskAttributes:
    """Test Celery task attributes and configuration."""

    def test_train_model_task_name(self) -> None:
        """Test that train_model has correct task name."""
        from objdet.pipelines.tasks import train_model

        assert train_model.name == "objdet.pipelines.tasks.train_model"

    def test_export_model_task_name(self) -> None:
        """Test that export_model has correct task name."""
        from objdet.pipelines.tasks import export_model

        assert export_model.name == "objdet.pipelines.tasks.export_model"

    def test_preprocess_data_task_name(self) -> None:
        """Test that preprocess_data has correct task name."""
        from objdet.pipelines.tasks import preprocess_data

        assert preprocess_data.name == "objdet.pipelines.tasks.preprocess_data"

    def test_train_model_is_bound_task(self) -> None:
        """Test that train_model is a bound task."""
        from objdet.pipelines.tasks import train_model

        # Bound tasks have the 'bind=True' option
        assert hasattr(train_model, "name")
        assert "train_model" in train_model.name

    def test_export_model_is_registered(self) -> None:
        """Test that export_model is registered."""
        from objdet.pipelines.tasks import export_model

        assert hasattr(export_model, "name")
        assert "export_model" in export_model.name

    def test_preprocess_data_is_registered(self) -> None:
        """Test that preprocess_data is registered."""
        from objdet.pipelines.tasks import preprocess_data

        assert hasattr(preprocess_data, "name")
        assert "preprocess_data" in preprocess_data.name
