"""Gradient monitoring callback.

This callback monitors gradient statistics during training,
useful for debugging training issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from lightning.pytorch.callbacks import Callback

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class GradientMonitorCallback(Callback):
    """Callback to monitor gradient statistics.

    Logs gradient norms, min/max values, and NaN/Inf detection
    for debugging training stability issues.

    Args:
        log_every_n_steps: How often to log gradient stats.
        detect_anomalies: Whether to enable anomaly detection.

    Example:
        >>> callback = GradientMonitorCallback(log_every_n_steps=50)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        detect_anomalies: bool = False,
    ) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.detect_anomalies = detect_anomalies

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: "Optimizer",
    ) -> None:
        """Monitor gradients before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = []
        grad_max = []
        has_nan = False
        has_inf = False

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                # Check for anomalies
                if grad.isnan().any():
                    has_nan = True
                    if self.detect_anomalies:
                        logger.warning(f"NaN gradient detected in {name}")

                if grad.isinf().any():
                    has_inf = True
                    if self.detect_anomalies:
                        logger.warning(f"Inf gradient detected in {name}")

                # Collect statistics
                grad_norm = grad.norm(2).item()
                grad_norms.append(grad_norm)
                grad_max.append(grad.abs().max().item())

        if grad_norms:
            import torch

            total_norm = torch.tensor(grad_norms).norm(2).item()
            max_grad = max(grad_max)
            mean_norm = sum(grad_norms) / len(grad_norms)

            # Log statistics
            pl_module.log("train/grad_norm", total_norm, prog_bar=False)
            pl_module.log("train/grad_max", max_grad, prog_bar=False)
            pl_module.log("train/grad_mean_norm", mean_norm, prog_bar=False)

            if has_nan:
                pl_module.log("train/grad_has_nan", 1.0, prog_bar=False)
            if has_inf:
                pl_module.log("train/grad_has_inf", 1.0, prog_bar=False)
