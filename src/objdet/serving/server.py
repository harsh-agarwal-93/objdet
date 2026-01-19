"""LitServe server implementation.

This module provides the server entry point for running
the detection API.

Example:
    >>> from objdet.serving import run_server
    >>>
    >>> run_server(
    ...     config_path="configs/serving/default.yaml",
    ...     host="0.0.0.0",
    ...     port=8000,
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from objdet.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def run_server(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers_per_device: int = 1,
    accelerator: str = "auto",
    devices: int | str = "auto",
    max_batch_size: int = 8,
    batch_timeout: float = 0.01,
    api_path: str = "/predict",
) -> None:
    """Run the detection inference server.

    Args:
        config_path: Path to serving configuration YAML.
        checkpoint_path: Path to model checkpoint (if not in config).
        host: Host to bind server.
        port: Port to bind server.
        workers_per_device: Number of worker processes per device.
        accelerator: Accelerator type ("auto", "cuda", "cpu").
        devices: Number of devices or "auto".
        max_batch_size: Maximum batch size for dynamic batching.
        batch_timeout: Timeout for batch collection (seconds).
        api_path: API endpoint path.
    """
    try:
        import litserve as ls
    except ImportError as e:
        raise RuntimeError("LitServe is required. Install with: pip install litserve") from e

    # Load config if provided
    config = {}
    if config_path:
        config = _load_config(Path(config_path))

    # Get checkpoint path
    ckpt_path = checkpoint_path or config.get("checkpoint_path")
    if not ckpt_path:
        raise ValueError("checkpoint_path is required (in config or as argument)")

    # Create API
    from objdet.serving.api import DetectionAPI

    api = DetectionAPI(
        checkpoint_path=ckpt_path,
        confidence_threshold=config.get("confidence_threshold", 0.25),
        max_batch_size=max_batch_size,
    )

    # Create server
    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path=api_path,
    )

    logger.info(f"Starting detection server on {host}:{port}")
    server.run(host=host, port=port)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)
