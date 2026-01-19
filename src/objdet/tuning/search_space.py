"""Search space definition for hyperparameter tuning.

This module provides utilities for defining hyperparameter search spaces
in a declarative way.

Example:
    >>> from objdet.tuning import define_search_space
    >>>
    >>> space = define_search_space(
    ...     learning_rate=("log_uniform", 1e-5, 1e-2),
    ...     batch_size=("categorical", [8, 16, 32]),
    ...     weight_decay=("uniform", 0.0, 0.1),
    ...     optimizer=("categorical", ["adam", "sgd", "adamw"]),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna

from objdet.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchSpace:
    """Hyperparameter search space definition.

    Attributes:
        params: Dictionary of parameter definitions.
    """

    params: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters from the search space.

        Args:
            trial: Optuna trial for sampling.

        Returns:
            Dictionary of sampled hyperparameters.
        """
        sampled = {}

        for name, spec in self.params.items():
            param_type = spec[0]

            if param_type == "uniform":
                low, high = spec[1], spec[2]
                sampled[name] = trial.suggest_float(name, low, high)

            elif param_type == "log_uniform":
                low, high = spec[1], spec[2]
                sampled[name] = trial.suggest_float(name, low, high, log=True)

            elif param_type == "int":
                low, high = spec[1], spec[2]
                step = spec[3] if len(spec) > 3 else 1
                sampled[name] = trial.suggest_int(name, low, high, step=step)

            elif param_type == "categorical":
                choices = spec[1]
                sampled[name] = trial.suggest_categorical(name, choices)

            elif param_type == "discrete_uniform":
                low, high, q = spec[1], spec[2], spec[3]
                # Use categorical for discrete choices
                choices = list(range(int(low), int(high) + 1, int(q)))
                sampled[name] = trial.suggest_categorical(name, choices)

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return sampled

    def add(
        self,
        name: str,
        param_type: str,
        *args: Any,
    ) -> SearchSpace:
        """Add a parameter to the search space.

        Args:
            name: Parameter name.
            param_type: Type of distribution.
            *args: Type-specific arguments.

        Returns:
            Self for chaining.
        """
        self.params[name] = (param_type,) + args
        return self


def define_search_space(**kwargs: tuple[str, ...]) -> SearchSpace:
    """Create a search space from keyword arguments.

    Each argument should be a tuple of (type, *args):
    - ("uniform", low, high): Continuous uniform
    - ("log_uniform", low, high): Log-uniform
    - ("int", low, high, [step]): Integer
    - ("categorical", [choices]): Categorical

    Args:
        **kwargs: Parameter definitions.

    Returns:
        SearchSpace instance.

    Example:
        >>> space = define_search_space(
        ...     lr=("log_uniform", 1e-5, 1e-2),
        ...     epochs=("int", 10, 100),
        ...     optimizer=("categorical", ["adam", "sgd"]),
        ... )
    """
    return SearchSpace(params=dict(kwargs))


# Common search space presets
DETECTION_SEARCH_SPACE = define_search_space(
    learning_rate=("log_uniform", 1e-5, 1e-2),
    weight_decay=("log_uniform", 1e-6, 1e-2),
    batch_size=("categorical", [4, 8, 16, 32]),
    optimizer=("categorical", ["adamw", "sgd"]),
    warmup_epochs=("int", 0, 5),
    label_smoothing=("uniform", 0.0, 0.1),
)

YOLO_SEARCH_SPACE = define_search_space(
    learning_rate=("log_uniform", 1e-4, 1e-1),
    momentum=("uniform", 0.8, 0.99),
    weight_decay=("log_uniform", 1e-5, 1e-2),
    warmup_epochs=("int", 0, 5),
    mosaic=("uniform", 0.5, 1.0),
    mixup=("uniform", 0.0, 0.5),
    hsv_h=("uniform", 0.0, 0.1),
    hsv_s=("uniform", 0.0, 0.9),
    hsv_v=("uniform", 0.0, 0.9),
)
