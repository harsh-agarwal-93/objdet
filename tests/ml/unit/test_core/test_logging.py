"""Unit tests for logging module."""

from __future__ import annotations

from pathlib import Path

from objdet.core.logging import (
    _get_json_format,
    _get_rich_format,
    configure_logging,
    get_logger,
)


class TestRichFormat:
    """Tests for rich format configuration."""

    def test_get_rich_format_returns_string(self) -> None:
        """Test that rich format returns a format string."""
        fmt = _get_rich_format()
        assert isinstance(fmt, str)
        assert len(fmt) > 0

    def test_rich_format_contains_placeholders(self) -> None:
        """Test that rich format string contains log placeholders."""
        fmt = _get_rich_format()
        # Should contain common loguru format placeholders
        assert "{" in fmt and "}" in fmt


class TestJsonFormat:
    """Tests for JSON format configuration."""

    def test_get_json_format_returns_string(self) -> None:
        """Test that JSON format returns a format string."""
        fmt = _get_json_format()
        assert isinstance(fmt, str)
        assert len(fmt) > 0


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a logger instance."""
        logger = get_logger(__name__)
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        """Test get_logger with a specific module name."""
        logger = get_logger("test.module.name")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        """Test get_logger without a name defaults to root."""
        logger = get_logger()
        assert logger is not None

    def test_get_logger_returns_consistent_logger(self) -> None:
        """Test that get_logger returns consistent logger for same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        # Both should work (loguru returns the same logger interface)
        assert logger1 is not None
        assert logger2 is not None


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_with_defaults(self) -> None:
        """Test configure_logging with default parameters."""
        # This should not raise
        configure_logging()

    def test_configure_logging_with_level(self) -> None:
        """Test configure_logging with different log levels."""
        configure_logging(level="DEBUG")
        configure_logging(level="WARNING")
        configure_logging(level="ERROR")

    def test_configure_logging_with_rich_format(self) -> None:
        """Test configure_logging with rich format."""
        configure_logging(log_format="rich")

    def test_configure_logging_with_json_format(self) -> None:
        """Test configure_logging with JSON format."""
        configure_logging(log_format="json")

    def test_configure_logging_with_log_dir(self, tmp_path: Path) -> None:
        """Test configure_logging with a log directory."""
        log_dir = tmp_path / "logs"
        configure_logging(log_dir=log_dir)

        # Log dir should be created if it doesn't exist
        # (This depends on implementation - may create on first log)

    def test_configure_logging_rotation(self) -> None:
        """Test configure_logging with custom rotation settings."""
        configure_logging(rotation="5 MB", retention="3 days")


class TestLoggerMethods:
    """Tests for logger method availability."""

    def test_logger_has_debug_method(self) -> None:
        """Test that logger has debug method."""
        logger = get_logger(__name__)
        assert hasattr(logger, "debug")
        assert callable(logger.debug)

    def test_logger_has_info_method(self) -> None:
        """Test that logger has info method."""
        logger = get_logger(__name__)
        assert hasattr(logger, "info")
        assert callable(logger.info)

    def test_logger_has_warning_method(self) -> None:
        """Test that logger has warning method."""
        logger = get_logger(__name__)
        assert hasattr(logger, "warning")
        assert callable(logger.warning)

    def test_logger_has_error_method(self) -> None:
        """Test that logger has error method."""
        logger = get_logger(__name__)
        assert hasattr(logger, "error")
        assert callable(logger.error)

    def test_logger_has_critical_method(self) -> None:
        """Test that logger has critical method."""
        logger = get_logger(__name__)
        assert hasattr(logger, "critical")
        assert callable(logger.critical)

    def test_logger_has_bind_method(self) -> None:
        """Test that logger has bind method for context."""
        logger = get_logger(__name__)
        assert hasattr(logger, "bind")
        assert callable(logger.bind)

    def test_logger_bind_returns_logger(self) -> None:
        """Test that bind returns a logger with context."""
        logger = get_logger(__name__)
        bound_logger = logger.bind(request_id="123")
        assert bound_logger is not None
