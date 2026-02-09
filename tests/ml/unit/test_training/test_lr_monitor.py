"""Unit tests for LearningRateMonitorCallback."""

from unittest.mock import MagicMock

import pytest

from objdet.training.callbacks.lr_monitor import LearningRateMonitorCallback


@pytest.fixture
def trainer() -> MagicMock:
    """Mock Lightning Trainer."""
    trainer = MagicMock()
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001}]
    trainer.optimizers = [optimizer]
    return trainer


@pytest.fixture
def pl_module() -> MagicMock:
    """Mock Lightning Module."""
    return MagicMock()


def test_init() -> None:
    """Test callback initialization."""
    cb = LearningRateMonitorCallback(log_momentum=True, log_weight_decay=True)
    assert cb.log_momentum is True
    assert cb.log_weight_decay is True


def test_on_train_batch_start_single_opt_single_pg(
    trainer: MagicMock, pl_module: MagicMock
) -> None:
    """Test logging with single optimizer and single parameter group."""
    cb = LearningRateMonitorCallback()
    cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)

    # Simple name "lr" should be used
    pl_module.log.assert_called_with("lr", 0.01, prog_bar=False)


def test_on_train_batch_start_multi_pg(trainer: MagicMock, pl_module: MagicMock) -> None:
    """Test logging with multiple parameter groups."""
    optimizer = trainer.optimizers[0]
    optimizer.param_groups = [
        {"lr": 0.01},
        {"lr": 0.02},
    ]

    cb = LearningRateMonitorCallback()
    cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)

    # Specific names should be used
    pl_module.log.assert_any_call("lr/optimizer_0_pg_0", 0.01, prog_bar=False)
    pl_module.log.assert_any_call("lr/optimizer_0_pg_1", 0.02, prog_bar=False)


def test_on_train_batch_start_multi_opt(trainer: MagicMock, pl_module: MagicMock) -> None:
    """Test logging with multiple optimizers."""
    opt1 = MagicMock()
    opt1.param_groups = [{"lr": 0.01}]
    opt2 = MagicMock()
    opt2.param_groups = [{"lr": 0.02}]
    trainer.optimizers = [opt1, opt2]

    cb = LearningRateMonitorCallback()
    cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)

    pl_module.log.assert_any_call("lr/optimizer_0_pg_0", 0.01, prog_bar=False)
    pl_module.log.assert_any_call("lr/optimizer_1_pg_0", 0.02, prog_bar=False)


def test_log_momentum_and_wd(trainer: MagicMock, pl_module: MagicMock) -> None:
    """Test logging momentum and weight decay."""
    param_group = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001}
    trainer.optimizers[0].param_groups = [param_group]

    cb = LearningRateMonitorCallback(log_momentum=True, log_weight_decay=True)
    cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)

    pl_module.log.assert_any_call("momentum/pg_0", 0.9, prog_bar=False)
    pl_module.log.assert_any_call("weight_decay/pg_0", 0.0001, prog_bar=False)


def test_log_momentum_betas(trainer: MagicMock, pl_module: MagicMock) -> None:
    """Test logging momentum from betas (Adam)."""
    param_group = {"lr": 0.01, "betas": (0.8, 0.999)}
    trainer.optimizers[0].param_groups = [param_group]

    cb = LearningRateMonitorCallback(log_momentum=True)
    cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)

    pl_module.log.assert_any_call("momentum/pg_0", 0.8, prog_bar=False)


def test_on_train_batch_start_no_optimizers(trainer: MagicMock, pl_module: MagicMock) -> None:
    """Test batch start when no optimizers are present."""
    trainer.optimizers = []
    cb = LearningRateMonitorCallback()
    cb.on_train_batch_start(trainer, pl_module, batch={}, batch_idx=0)
    assert not pl_module.log.called
