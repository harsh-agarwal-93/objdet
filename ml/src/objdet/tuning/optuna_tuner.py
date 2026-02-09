"""Optuna hyperparameter tuner for object detection.

This module provides integration with Optuna for automated
hyperparameter optimization.

Example:
    >>> from objdet.tuning import OptunaTuner, define_search_space
    >>>
    >>> search_space = define_search_space(
    ...     lr=("log_uniform", 1e-5, 1e-2),
    ...     batch_size=("categorical", [8, 16, 32]),
    ... )
    >>> tuner = OptunaTuner(
    ...     search_space=search_space,
    ...     n_trials=50,
    ...     study_name="coco_frcnn_tuning",
    ... )
    >>> best_params = tuner.run(
    ...     config_path="configs/base.yaml",
    ...     train_fn=train_and_evaluate,
    ... )
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    import lightning as L

    from objdet.tuning.search_space import SearchSpace

logger = get_logger(__name__)


class OptunaTuner:
    """Optuna-based hyperparameter tuner.

    Args:
        search_space: Search space definition from define_search_space.
        n_trials: Number of trials to run.
        study_name: Name for the Optuna study.
        storage: Optuna storage URL (for distributed tuning).
        direction: Optimization direction ("maximize" or "minimize").
        pruner: Optuna pruner (default: MedianPruner).
        sampler: Optuna sampler (default: TPE).
        timeout: Maximum time in seconds.

    Example:
        >>> tuner = OptunaTuner(search_space, n_trials=100)
        >>> best = tuner.run(config_path, train_fn)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 50,
        study_name: str = "objdet_tuning",
        storage: str | None = None,
        direction: str = "maximize",
        pruner: optuna.pruners.BasePruner | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        timeout: int | None = None,
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.timeout = timeout

        self.pruner = pruner or MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        self.sampler = sampler or TPESampler(seed=42)

        self._study: optuna.Study | None = None

    def run(
        self,
        config_path: str | Path,
        train_fn: Callable[[dict[str, Any], str | Path], float],
        output_dir: str | Path = "outputs/tuning",
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            config_path: Base configuration file.
            train_fn: Training function that takes (params, config_path)
                      and returns the metric to optimize.
            output_dir: Directory for outputs.

        Returns:
            Best hyperparameters found.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create or load study
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self.search_space.sample(trial)

            logger.info(f"Trial {trial.number}: {params}")

            # Run training
            try:
                metric = train_fn(params, config_path)
                return metric
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()

        # Run optimization
        logger.info(
            f"Starting Optuna optimization: {self.n_trials} trials, direction={self.direction}"
        )

        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        # Get best results
        best_trial = self._study.best_trial
        best_params = best_trial.params

        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value}")
        logger.info(f"Best params: {best_params}")

        # Save results
        self._save_results(output_dir)

        return best_params

    def _save_results(self, output_dir: Path) -> None:
        """Save tuning results to files."""
        if not self._study:
            return

        import json

        # Save best parameters
        best_params_path = output_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "params": self._study.best_params,
                    "value": self._study.best_value,
                    "trial_number": self._study.best_trial.number,
                },
                f,
                indent=2,
            )

        # Save all trials
        trials_path = output_dir / "trials.json"
        trials_data = []
        for trial in self._study.trials:
            trials_data.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
            )

        with open(trials_path, "w") as f:
            json.dump(trials_data, f, indent=2)

        logger.info(f"Saved tuning results to {output_dir}")

    @property
    def study(self) -> optuna.Study | None:
        """Get the Optuna study."""
        return self._study


class PyTorchLightningPruningCallback:
    """Optuna pruning callback for PyTorch Lightning.

    Reports intermediate values and prunes unpromising trials.

    Args:
        trial: Optuna trial object.
        monitor: Metric to monitor.

    Example:
        >>> callback = PyTorchLightningPruningCallback(trial, "val/mAP")
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val/mAP") -> None:
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(
        self,
        trainer: L.Trainer,
    ) -> None:
        """Report and possibly prune after validation."""
        epoch = trainer.current_epoch
        current_value = trainer.callback_metrics.get(self.monitor)

        if current_value is None:
            return

        # Report to Optuna
        self.trial.report(float(current_value), epoch)

        # Check if should prune
        if self.trial.should_prune():
            raise optuna.TrialPruned()
