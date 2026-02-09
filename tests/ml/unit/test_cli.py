"""Unit tests for the ObjDet CLI."""

from unittest.mock import MagicMock, patch

import pytest

from objdet.cli import ObjDetCLI, create_cli, main


@pytest.fixture
def mock_parser() -> MagicMock:
    """Mock LightningArgumentParser."""
    return MagicMock()


@patch("objdet.cli.LightningCLI.__init__", return_value=None)
def test_create_cli(mock_cli_init: MagicMock) -> None:
    """Test create_cli factory function."""
    create_cli()
    assert mock_cli_init.called


def test_objdet_cli_add_arguments(mock_parser: MagicMock) -> None:
    """Test custom argument adding."""
    # We need a proper instance, but ObjDetCLI(LightningCLI) calls __init__ which is heavy.
    # Let's mock __init__.
    with patch("lightning.pytorch.cli.LightningCLI.__init__", return_value=None):
        cli = ObjDetCLI()
        cli.add_arguments_to_parser(mock_parser)
        assert mock_parser.add_argument.called


def test_objdet_cli_hooks() -> None:
    """Test CLI lifecycle hooks."""
    with patch("lightning.pytorch.cli.LightningCLI.__init__", return_value=None):
        cli = ObjDetCLI()
        cli.config = MagicMock()
        cli.config.subcommand = "fit"
        cli.config.fit = MagicMock()

        with patch("objdet.cli.configure_logging") as mock_log:
            cli.before_instantiate_classes()
            assert mock_log.called

        # Test other hooks (just coverage)
        cli.after_fit()
        cli.after_validate()
        cli.after_test()
        cli.after_predict()


def test_main_subcommands() -> None:
    """Test main entry point for custom subcommands."""
    # Serve
    with (
        patch("sys.argv", ["objdet", "serve", "--config", "test.yaml"]),
        patch("objdet.cli._run_serve_command") as mock_run,
    ):
        main()
        mock_run.assert_called_once()

    # Export
    with (
        patch("sys.argv", ["objdet", "export", "--checkpoint", "c.ckpt", "--output", "o.onnx"]),
        patch("objdet.cli._run_export_command") as mock_run,
    ):
        main()
        mock_run.assert_called_once()

    # Preprocess
    with (
        patch("sys.argv", ["objdet", "preprocess", "--output", "out", "--format", "coco"]),
        patch("objdet.cli._run_preprocess_command") as mock_run,
    ):
        main()
        mock_run.assert_called_once()


def test_main_lightning_cli() -> None:
    """Test main falls back to standard Lightning CLI."""
    with patch("sys.argv", ["objdet", "fit"]), patch("objdet.cli.create_cli") as mock_create:
        main()
        mock_create.assert_called_once()


@patch("objdet.serving.server.run_server")
def test_run_serve_command(mock_run_server: MagicMock) -> None:
    """Test internal serve runner."""
    from objdet.cli import _run_serve_command

    with patch("sys.argv", ["objdet", "serve", "--config", "test.yaml", "--port", "9000"]):
        _run_serve_command()
        mock_run_server.assert_called_once()


@patch("objdet.optimization.export_model")
def test_run_export_command(mock_export: MagicMock) -> None:
    """Test internal export runner."""
    from objdet.cli import _run_export_command

    with patch(
        "sys.argv",
        [
            "objdet",
            "export",
            "--checkpoint",
            "c.ckpt",
            "--output",
            "o.onnx",
            "--format",
            "safetensors",
        ],
    ):
        _run_export_command()
        mock_export.assert_called_once()


@patch("objdet.data.preprocessing.convert_to_litdata")
def test_run_preprocess_command(mock_convert: MagicMock) -> None:
    """Test internal preprocess runner."""
    from objdet.cli import _run_preprocess_command

    with patch(
        "sys.argv",
        [
            "objdet",
            "preprocess",
            "--output",
            "out",
            "--format",
            "coco",
            "--coco_ann_file",
            "a.json",
        ],
    ):
        _run_preprocess_command()
        mock_convert.assert_called_once()
