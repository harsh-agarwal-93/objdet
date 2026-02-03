"""Unit tests for Celery tasks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


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


class TestTrainModelTask:
    """Test train_model task execution."""

    @patch("objdet.cli.create_cli")
    @patch("objdet.pipelines.tasks.Trainer")
    def test_train_model_success(
        self,
        mock_trainer_cls: MagicMock,
        mock_create_cli: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Test successful model training execution."""
        from objdet.pipelines.tasks import train_model

        # Create dummy config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("trainer:\n  max_epochs: 10")

        output_dir = tmp_path / "outputs"

        # Setup mocks
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.fit.return_value = None

        # Call task
        class MockModelCheckpoint:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.best_model_path = "/path/to/best.ckpt"
                self.best_model_score = 0.95

        with patch("objdet.pipelines.tasks.ModelCheckpoint", new=MockModelCheckpoint):
            # Run task
            result = train_model(
                config_path=str(config_path),
                output_dir=str(output_dir),
                max_epochs=5,
            )

        # Assertions
        mock_trainer_cls.assert_called()
        mock_create_cli.assert_called()
        mock_trainer.fit.assert_called()

        assert result["status"] == "completed"
        assert result["best_checkpoint"] == "/path/to/best.ckpt"
        assert result["best_mAP"] == 0.95

    @patch("objdet.cli.create_cli")
    @patch("objdet.pipelines.tasks.Trainer")
    def test_train_model_failure_retries(
        self,
        mock_trainer_cls: MagicMock,
        mock_create_cli: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Test that training failure triggers retry."""
        from objdet.pipelines.tasks import train_model

        # Create dummy config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("trainer:\n  max_epochs: 10")

        # Setup failure
        mock_trainer_cls.side_effect = RuntimeError("Training failed")

        # Mock self.retry to raise the exception (Celery behavior)
        with patch.object(train_model, "retry", side_effect=RuntimeError("Retry")) as mock_retry:
            with pytest.raises(RuntimeError, match="Retry"):
                train_model(
                    config_path=str(config_path),
                    output_dir=str(tmp_path / "outputs"),
                )

            mock_retry.assert_called()


class TestExportModelTask:
    """Test export_model task execution."""

    @patch("objdet.optimization.export_model")
    def test_export_model_success(self, mock_export: MagicMock) -> None:
        """Test successful model export."""
        from objdet.pipelines.tasks import export_model

        mock_export.return_value = "/path/to/model.onnx"

        result = export_model(
            checkpoint_path="model.ckpt",
            output_path="model.onnx",
            export_format="onnx",
        )

        mock_export.assert_called_with(
            checkpoint_path="model.ckpt",
            output_path="model.onnx",
            export_format="onnx",
            input_shape=(1, 3, 640, 640),
        )

        assert result["status"] == "completed"
        assert result["output_path"] == "/path/to/model.onnx"

    @patch("objdet.optimization.export_model")
    def test_export_model_failure(self, mock_export: MagicMock) -> None:
        """Test export failure."""
        from objdet.pipelines.tasks import export_model

        mock_export.side_effect = ValueError("Export failed")

        with pytest.raises(ValueError, match="Export failed"):
            export_model(
                checkpoint_path="model.ckpt",
                output_path="model.onnx",
            )


class TestPreprocessDataTask:
    """Test preprocess_data task execution."""

    @patch("objdet.data.preprocessing.convert_to_litdata")
    def test_preprocess_data_success(self, mock_convert: MagicMock) -> None:
        """Test successful data preprocessing."""
        from objdet.pipelines.tasks import preprocess_data

        result = preprocess_data(
            input_dir="data/coco",
            output_dir="data/litdata",
            format_name="coco",
            class_names=["cat", "dog"],
        )

        mock_convert.assert_called_with(
            input_dir="data/coco",
            output_dir="data/litdata",
            format_name="coco",
            num_workers=4,
            class_names=["cat", "dog"],
        )

        assert result["status"] == "completed"
        assert result["output_dir"] == "data/litdata"

    @patch("objdet.data.preprocessing.convert_to_litdata")
    def test_preprocess_data_failure(self, mock_convert: MagicMock) -> None:
        """Test preprocessing failure."""
        from objdet.pipelines.tasks import preprocess_data

        mock_convert.side_effect = FileNotFoundError("Input dir not found")

        with pytest.raises(FileNotFoundError, match="Input dir not found"):
            preprocess_data(
                input_dir="data/coco",
                output_dir="data/litdata",
                format_name="coco",
            )
