"""Centralized logging configuration for ObjDet.

This module provides a unified logging setup using loguru with rich console output.
It supports log rotation, structured JSON logging for production, and integration
with Lightning's logging system.

Example:
    >>> from objdet.core.logging import configure_logging, get_logger
    >>>
    >>> # Configure at application startup
    >>> configure_logging(level="DEBUG", log_format="rich")
    >>>
    >>> # Get a logger for your module
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started", extra={"epochs": 100})
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Remove default loguru handler
logger.remove()

# Module-level flag to track if logging has been configured
_configured = False


def _get_rich_format() -> str:
    """Get the format string for rich console output."""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )


def _get_json_format() -> str:
    """Get the format string for JSON structured logging."""
    return "{message}"


def _json_serializer(record: dict[str, Any]) -> str:
    """Serialize log record to JSON for structured logging.

    Args:
        record: Loguru log record.

    Returns:
        JSON-formatted string.
    """
    import json

    from whenever import Instant

    log_entry = {
        "timestamp": str(Instant.now()),
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add extra fields
    if record.get("extra"):
        log_entry["extra"] = record["extra"]

    # Add exception info if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
            "traceback": record["exception"].traceback is not None,
        }

    return json.dumps(log_entry)


def configure_logging(
    level: str = "INFO",
    log_format: str = "rich",
    log_dir: Path | str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "gz",
) -> None:
    """Configure centralized logging for the application.

    This function should be called once at application startup to set up
    logging consistently across all modules.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - "rich" for colored console, "json" for structured.
        log_dir: Directory for log files. If None, only console logging is enabled.
        rotation: When to rotate log files (e.g., "10 MB", "1 day").
        retention: How long to keep old log files (e.g., "7 days", "1 month").
        compression: Compression format for rotated files (e.g., "gz", "zip").

    Example:
        >>> configure_logging(
        ...     level="DEBUG",
        ...     log_format="rich",
        ...     log_dir="./logs",
        ...     rotation="100 MB",
        ...     retention="30 days",
        ... )
    """
    global _configured  # noqa: PLW0603

    # Remove any existing handlers
    logger.remove()

    # Get level from environment or use provided
    level = os.environ.get("OBJDET_LOG_LEVEL", level).upper()
    log_format = os.environ.get("OBJDET_LOG_FORMAT", log_format).lower()

    # Console handler
    if log_format == "json":
        logger.add(
            sys.stderr,
            format=_get_json_format(),
            level=level,
            serialize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        # Rich format (default)
        logger.add(
            sys.stderr,
            format=_get_rich_format(),
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File handler with rotation
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Regular log file
        logger.add(
            log_path / "objdet_{time:YYYY-MM-DD}.log",
            format=_get_json_format() if log_format == "json" else _get_rich_format(),
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=log_format == "json",
            backtrace=True,
            diagnose=True,
        )

        # Separate error log
        logger.add(
            log_path / "objdet_errors_{time:YYYY-MM-DD}.log",
            format=_get_json_format() if log_format == "json" else _get_rich_format(),
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=log_format == "json",
            backtrace=True,
            diagnose=True,
        )

    _configured = True
    logger.debug(f"Logging configured: level={level}, format={log_format}")


def get_logger(name: str | None = None) -> LoggerInterface:
    """Get a logger instance for the specified module.

    This returns a loguru logger that can be used for logging throughout
    the application. If logging hasn't been configured, a default
    configuration will be applied.

    Args:
        name: Module name, typically __name__. Used for log context.

    Returns:
        Logger instance with the module name bound.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Processing item", item_id=123)
    """
    global _configured

    if not _configured:
        configure_logging()

    if name:
        return logger.bind(name=name)
    return logger


class LoggerInterface:
    """Type hint interface for the logger returned by get_logger.

    This class is not instantiated directly; it serves as documentation
    for the available logging methods.
    """

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        ...

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        ...

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        ...

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        ...

    def bind(self, **kwargs: Any) -> LoggerInterface:
        """Bind context to the logger."""
        ...


def intercept_standard_logging() -> None:
    """Intercept standard library logging and redirect to loguru.

    This is useful for capturing logs from third-party libraries
    that use the standard logging module.

    Example:
        >>> intercept_standard_logging()
        >>> # Now all logging.* calls go through loguru
    """
    import logging

    class InterceptHandler(logging.Handler):
        """Handler that redirects standard logging to loguru."""

        def emit(self, record: logging.LogRecord) -> None:
            """Emit a log record."""
            # Get corresponding loguru level
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where the logging call originated
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Remove existing handlers and add intercept handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def log_training_start(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    **extra: Any,
) -> None:
    """Log training start with structured metadata.

    Args:
        model_name: Name of the model being trained.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        **extra: Additional metadata to log.
    """
    logger.info(
        "Training started",
        model=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **extra,
    )


def log_training_end(
    model_name: str,
    final_loss: float,
    best_metric: float,
    duration_seconds: float,
    **extra: Any,
) -> None:
    """Log training completion with results.

    Args:
        model_name: Name of the model trained.
        final_loss: Final training loss.
        best_metric: Best validation metric achieved.
        duration_seconds: Total training duration.
        **extra: Additional metadata to log.
    """
    logger.info(
        "Training completed",
        model=model_name,
        final_loss=final_loss,
        best_metric=best_metric,
        duration_seconds=duration_seconds,
        **extra,
    )
