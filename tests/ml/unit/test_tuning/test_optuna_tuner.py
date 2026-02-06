"""Unit tests for OptunaTuner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from objdet.tuning.optuna_tuner import OptunaTuner


class TestOptunaTuner:
    """Tests for OptunaTuner class."""

    @pytest.fixture
    def mock_search_space(self) -> MagicMock:
        """Create a mock search space."""
        search_space = MagicMock()
        search_space.sample = MagicMock(return_value={"lr": 0.001, "batch_size": 16})
        return search_space

    def test_init_default_params(self, mock_search_space: MagicMock) -> None:
        """Test initialization with default parameters."""
        tuner = OptunaTuner(search_space=mock_search_space)

        assert tuner.n_trials == 50
        assert tuner.study_name == "objdet_tuning"
        assert tuner.direction == "maximize"
        assert tuner.timeout is None
        assert tuner._study is None

    def test_init_custom_params(self, mock_search_space: MagicMock) -> None:
        """Test initialization with custom parameters."""
        tuner = OptunaTuner(
            search_space=mock_search_space,
            n_trials=100,
            study_name="custom_study",
            direction="minimize",
            timeout=3600,
        )

        assert tuner.n_trials == 100
        assert tuner.study_name == "custom_study"
        assert tuner.direction == "minimize"
        assert tuner.timeout == 3600

    def test_init_default_pruner(self, mock_search_space: MagicMock) -> None:
        """Test default MedianPruner is created."""
        from optuna.pruners import MedianPruner

        tuner = OptunaTuner(search_space=mock_search_space)

        assert isinstance(tuner.pruner, MedianPruner)

    def test_init_default_sampler(self, mock_search_space: MagicMock) -> None:
        """Test default TPESampler is created."""
        from optuna.samplers import TPESampler

        tuner = OptunaTuner(search_space=mock_search_space)

        assert isinstance(tuner.sampler, TPESampler)

    def test_init_custom_pruner(self, mock_search_space: MagicMock) -> None:
        """Test custom pruner is used."""
        from optuna.pruners import NopPruner

        custom_pruner = NopPruner()
        tuner = OptunaTuner(search_space=mock_search_space, pruner=custom_pruner)

        assert tuner.pruner is custom_pruner

    def test_study_property_before_run(self, mock_search_space: MagicMock) -> None:
        """Test study property returns None before run."""
        tuner = OptunaTuner(search_space=mock_search_space)

        assert tuner.study is None

    def test_run_creates_study(self, mock_search_space: MagicMock, tmp_path: Path) -> None:
        """Test that run creates an Optuna study."""
        tuner = OptunaTuner(
            search_space=mock_search_space,
            n_trials=1,
        )

        def mock_train_fn(params, config_path):
            return 0.85

        with patch("optuna.create_study") as mock_create:
            mock_study = MagicMock()
            mock_study.best_trial.number = 0
            mock_study.best_trial.value = 0.85
            mock_study.best_trial.params = {"lr": 0.001}
            mock_study.best_params = {"lr": 0.001}
            mock_study.best_value = 0.85
            mock_study.trials = []
            mock_create.return_value = mock_study

            result = tuner.run(
                config_path=tmp_path / "config.yaml",
                train_fn=mock_train_fn,
                output_dir=tmp_path / "output",
            )

            mock_create.assert_called_once()
            assert result == {"lr": 0.001}

    def test_save_results(self, mock_search_space: MagicMock, tmp_path: Path) -> None:
        """Test that results are saved correctly."""
        import json

        tuner = OptunaTuner(search_space=mock_search_space, n_trials=1)

        # Create a mock study
        mock_study = MagicMock()
        mock_study.best_params = {"lr": 0.001}
        mock_study.best_value = 0.85
        mock_study.best_trial.number = 0
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.value = 0.85
        mock_trial.params = {"lr": 0.001}
        mock_trial.state = MagicMock()
        mock_trial.state.name = "COMPLETE"
        mock_study.trials = [mock_trial]

        tuner._study = mock_study

        output_dir = tmp_path / "tuning_output"
        output_dir.mkdir()
        tuner._save_results(output_dir)

        # Check files were created
        best_params_path = output_dir / "best_params.json"
        trials_path = output_dir / "trials.json"

        assert best_params_path.exists()
        assert trials_path.exists()

        # Verify content
        with open(best_params_path) as f:
            best_data = json.load(f)
            assert best_data["params"] == {"lr": 0.001}
            assert best_data["value"] == 0.85


class TestPyTorchLightningPruningCallback:
    """Tests for PyTorchLightningPruningCallback."""

    def test_init(self) -> None:
        """Test callback initialization."""
        from objdet.tuning.optuna_tuner import PyTorchLightningPruningCallback

        mock_trial = MagicMock()
        callback = PyTorchLightningPruningCallback(trial=mock_trial, monitor="val/mAP")

        assert callback.trial is mock_trial
        assert callback.monitor == "val/mAP"

    def test_init_default_monitor(self) -> None:
        """Test callback with default monitor."""
        from objdet.tuning.optuna_tuner import PyTorchLightningPruningCallback

        mock_trial = MagicMock()
        callback = PyTorchLightningPruningCallback(trial=mock_trial)

        assert callback.monitor == "val/mAP"

    def test_on_validation_end_reports_value(self) -> None:
        """Test that validation end reports value to trial."""
        from objdet.tuning.optuna_tuner import PyTorchLightningPruningCallback

        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False
        callback = PyTorchLightningPruningCallback(trial=mock_trial, monitor="val/loss")

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.callback_metrics = {"val/loss": 0.5}

        mock_pl_module = MagicMock()

        callback.on_validation_end(mock_trainer, mock_pl_module)

        mock_trial.report.assert_called_once_with(0.5, 5)

    def test_on_validation_end_handles_missing_metric(self) -> None:
        """Test that missing metric doesn't cause error."""
        from objdet.tuning.optuna_tuner import PyTorchLightningPruningCallback

        mock_trial = MagicMock()
        callback = PyTorchLightningPruningCallback(trial=mock_trial, monitor="val/mAP")

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        mock_trainer.callback_metrics = {}  # No metrics

        mock_pl_module = MagicMock()

        # Should not raise
        callback.on_validation_end(mock_trainer, mock_pl_module)

        # Should not have reported anything
        mock_trial.report.assert_not_called()
