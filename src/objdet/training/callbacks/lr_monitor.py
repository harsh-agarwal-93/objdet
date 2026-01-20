"""Learning rate monitor callback.

Enhanced learning rate logging with scheduler information.
"""

from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback

from objdet.core.logging import get_logger

logger = get_logger(__name__)


class LearningRateMonitorCallback(Callback):
    """Callback to monitor and log learning rates.

    This extends Lightning's built-in LearningRateMonitor with
    additional logging for multiple parameter groups.

    Args:
        log_momentum: Whether to also log momentum values.
        log_weight_decay: Whether to log weight decay values.

    Example:
        >>> callback = LearningRateMonitorCallback(log_momentum=True)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
    ) -> None:
        super().__init__()
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log learning rates at batch start."""
        if not trainer.optimizers:
            return

        for opt_idx, optimizer in enumerate(trainer.optimizers):
            for pg_idx, param_group in enumerate(optimizer.param_groups):
                # Learning rate
                lr = param_group.get("lr", 0)
                name = f"lr/optimizer_{opt_idx}_pg_{pg_idx}"
                if len(trainer.optimizers) == 1 and len(optimizer.param_groups) == 1:
                    name = "lr"
                pl_module.log(name, lr, prog_bar=False)

                # Momentum
                if self.log_momentum:
                    momentum = param_group.get("momentum", param_group.get("betas", (0,))[0])
                    pl_module.log(f"momentum/pg_{pg_idx}", momentum, prog_bar=False)

                # Weight decay
                if self.log_weight_decay:
                    wd = param_group.get("weight_decay", 0)
                    pl_module.log(f"weight_decay/pg_{pg_idx}", wd, prog_bar=False)
