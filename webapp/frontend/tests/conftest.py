"""Pytest configuration and fixtures for frontend tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def backend_url() -> str:
    """Backend URL for tests.

    Returns:
        Backend URL string.
    """
    return "http://localhost:8000"


@pytest.fixture
def sample_metrics_df() -> pl.DataFrame:
    """Sample metrics DataFrame for testing.

    Returns:
        Polars DataFrame with sample metrics data.
    """
    return pl.DataFrame(
        {
            "step": [0, 1, 2, 0, 1, 2],
            "metric": ["loss", "loss", "loss", "accuracy", "accuracy", "accuracy"],
            "value": [1.0, 0.8, 0.5, 0.7, 0.8, 0.9],
            "timestamp": [1000, 1001, 1002, 1000, 1001, 1002],
        }
    )


@pytest.fixture
def mock_streamlit(mocker: MockerFixture) -> dict[str, Any]:
    """Mock Streamlit functions for component testing.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Dictionary of mocked Streamlit functions.
    """
    return {
        "markdown": mocker.patch("streamlit.markdown"),
        "info": mocker.patch("streamlit.info"),
        "plotly_chart": mocker.patch("streamlit.plotly_chart"),
        "container": mocker.patch("streamlit.container"),
        "columns": mocker.patch("streamlit.columns"),
        "button": mocker.patch("streamlit.button"),
        "write": mocker.patch("streamlit.write"),
        "caption": mocker.patch("streamlit.caption"),
        "subheader": mocker.patch("streamlit.subheader"),
        "progress": mocker.patch("streamlit.progress"),
        "error": mocker.patch("streamlit.error"),
        "rerun": mocker.patch("streamlit.rerun"),
        "divider": mocker.patch("streamlit.divider"),
    }
