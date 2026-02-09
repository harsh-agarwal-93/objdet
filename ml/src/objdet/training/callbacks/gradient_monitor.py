"""Gradient monitoring callback.

This callback monitors gradient statistics during training,
useful for debugging training issues.
"""

from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer

from objdet.core.logging import get_logger

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
        optimizer: Optimizer,
    ) -> None:
        """Monitor gradients before optimizer step."""
        grad_norms, grad_max, has_nan, has_inf = self._process_gradients(pl_module)

        if grad_norms:
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

    def _process_gradients(
        self, pl_module: L.LightningModule
    ) -> tuple[list[float], list[float], bool, bool]:
        """Process gradients and collect statistics."""
        grad_norms = []
        grad_max = []
        has_nan = False
        has_inf = False

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            is_nan, is_inf = self._check_anomalies(grad, name)
            has_nan = has_nan or is_nan
            has_inf = has_inf or is_inf

            # Collect statistics
            grad_norms.append(grad.norm(2).item())
            grad_max.append(grad.abs().max().item())

        return grad_norms, grad_max, has_nan, has_inf

    def _check_anomalies(self, grad: torch.Tensor, name: str) -> tuple[bool, bool]:
        """Check for NaN and Inf in gradients."""
        is_nan = bool(grad.isnan().any())
        is_inf = bool(grad.isinf().any())

        if self.detect_anomalies:
            if is_nan:
                logger.warning(f"NaN gradient detected in {name}")
            if is_inf:
                logger.warning(f"Inf gradient detected in {name}")

        return is_nan, is_inf
