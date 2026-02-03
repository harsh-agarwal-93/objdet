"""Utility functions for data formatting."""

from __future__ import annotations

from datetime import datetime


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string (e.g., "4h 32m").
    """
    if seconds < 60:
        return f"{int(seconds)}s"

    minutes = int(seconds / 60)
    if minutes < 60:
        return f"{minutes}m"

    hours = minutes // 60
    remaining_mins = minutes % 60
    return f"{hours}h {remaining_mins}m"


def format_timestamp(timestamp_ms: int) -> str:
    """Format Unix timestamp in milliseconds to readable date.

    Args:
        timestamp_ms: Unix timestamp in milliseconds.

    Returns:
        Formatted date string.
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_iso_timestamp(iso_string: str) -> str:
    """Format ISO 8601 timestamp to readable string.

    Args:
        iso_string: ISO 8601 timestamp string.

    Returns:
        Formatted date string.
    """
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_string


def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string.

    Args:
        bytes_size: File size in bytes.

    Returns:
        Formatted file size (e.g., "45.6 MB").
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024**2:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size / (1024**2):.1f} MB"
    else:
        return f"{bytes_size / (1024**3):.1f} GB"


def format_metric(value: float, decimals: int = 2) -> str:
    """Format metric value.

    Args:
        value: Metric value.
        decimals: Number of decimal places.

    Returns:
        Formatted metric string.
    """
    return f"{value:.{decimals}f}"
