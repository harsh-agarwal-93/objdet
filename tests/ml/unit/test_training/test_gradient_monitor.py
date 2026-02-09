from unittest.mock import MagicMock

import lightning as L
import pytest
import torch

from objdet.training.callbacks.gradient_monitor import GradientMonitorCallback


def test_gradient_monitor_callback_basic():
    callback = GradientMonitorCallback(log_every_n_steps=1, detect_anomalies=True)

    trainer = MagicMock(spec=L.Trainer)
    pl_module = MagicMock(spec=L.LightningModule)
    optimizer = MagicMock()

    # Mock parameters
    param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    param1.grad = torch.tensor([0.1, 0.2])

    pl_module.named_parameters.return_value = [
        ("param1", param1),
    ]

    callback.on_before_optimizer_step(trainer, pl_module, optimizer)

    # Verify logging calls
    # expected norms: [sqrt(0.1^2 + 0.2^2) = 0.2236]
    pl_module.log.assert_any_call("train/grad_norm", pytest.approx(0.2236, 0.001), prog_bar=False)
    pl_module.log.assert_any_call("train/grad_max", pytest.approx(0.2, 0.001), prog_bar=False)


def test_gradient_monitor_callback_nan():
    callback = GradientMonitorCallback(log_every_n_steps=1, detect_anomalies=True)

    trainer = MagicMock(spec=L.Trainer)
    pl_module = MagicMock(spec=L.LightningModule)
    optimizer = MagicMock()

    # Mock parameters
    param1 = torch.nn.Parameter(torch.tensor([1.0]))
    param1.grad = torch.tensor([float("nan")])

    pl_module.named_parameters.return_value = [
        ("param1", param1),
    ]

    callback.on_before_optimizer_step(trainer, pl_module, optimizer)

    pl_module.log.assert_any_call("train/grad_has_nan", 1.0, prog_bar=False)
