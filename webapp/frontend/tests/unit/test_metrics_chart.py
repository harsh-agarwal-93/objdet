"""Tests for metrics chart component."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_st_plotly_chart(mocker: MockerFixture) -> MockerFixture:
    """Mock streamlit.plotly_chart.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Mocked plotly_chart function.
    """
    return mocker.patch("frontend.components.metrics_chart.st.plotly_chart")


@pytest.fixture
def mock_st_info(mocker: MockerFixture) -> MockerFixture:
    """Mock streamlit.info.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Mocked info function.
    """
    return mocker.patch("frontend.components.metrics_chart.st.info")


def test_plot_metrics_chart_with_data(
    mock_st_plotly_chart: MockerFixture, sample_metrics_df: pl.DataFrame
) -> None:
    """Test metrics chart with valid data.

    Args:
        mock_st_plotly_chart: Mocked plotly chart function.
        sample_metrics_df: Sample metrics DataFrame.
    """
    from frontend.components.metrics_chart import plot_metrics_chart

    plot_metrics_chart(sample_metrics_df, title="Test Metrics")

    # Verify plotly chart was called
    assert mock_st_plotly_chart.called

    # Get the figure that was passed
    fig = mock_st_plotly_chart.call_args[0][0]

    # Verify figure has data
    assert len(fig.data) == 2  # loss and accuracy metrics

    # Verify title
    assert fig.layout.title.text == "Test Metrics"


def test_plot_metrics_chart_empty(
    mock_st_info: MockerFixture, mock_st_plotly_chart: MockerFixture
) -> None:
    """Test metrics chart with empty DataFrame shows info.

    Args:
        mock_st_info: Mocked info function.
        mock_st_plotly_chart: Mocked plotly chart function.
    """
    from frontend.components.metrics_chart import plot_metrics_chart

    empty_df = pl.DataFrame(schema={"step": pl.Int64, "metric": pl.Utf8, "value": pl.Float64})
    plot_metrics_chart(empty_df)

    # Should show info message
    assert mock_st_info.called
    assert "No metrics data available" in mock_st_info.call_args[0][0]

    # Should not create chart
    assert not mock_st_plotly_chart.called


def test_plot_loss_curves(
    mock_st_plotly_chart: MockerFixture, sample_metrics_df: pl.DataFrame
) -> None:
    """Test loss curves filtering and plotting.

    Args:
        mock_st_plotly_chart: Mocked plotly chart function.
        sample_metrics_df: Sample metrics DataFrame.
    """
    from frontend.components.metrics_chart import plot_loss_curves

    plot_loss_curves(sample_metrics_df)

    # Verify plotly chart was called
    assert mock_st_plotly_chart.called

    # Get the figure
    fig = mock_st_plotly_chart.call_args[0][0]

    # Should only have loss metric (1 trace)
    assert len(fig.data) == 1
    assert fig.data[0].name == "loss"

    # Verify title
    assert fig.layout.title.text == "Loss Curves"


def test_plot_loss_curves_no_data(mock_st_info: MockerFixture) -> None:
    """Test loss curves with no loss metrics shows info.

    Args:
        mock_st_info: Mocked info function.
    """
    from frontend.components.metrics_chart import plot_loss_curves

    # DataFrame with no loss metrics
    df = pl.DataFrame(
        {
            "step": [0, 1, 2],
            "metric": ["accuracy", "accuracy", "accuracy"],
            "value": [0.7, 0.8, 0.9],
            "timestamp": [1000, 1001, 1002],
        }
    )

    plot_loss_curves(df)

    # Should show info message
    assert mock_st_info.called
    assert "No loss metrics available" in mock_st_info.call_args[0][0]


def test_plot_accuracy_curves(
    mock_st_plotly_chart: MockerFixture, sample_metrics_df: pl.DataFrame
) -> None:
    """Test accuracy curves filtering and plotting.

    Args:
        mock_st_plotly_chart: Mocked plotly chart function.
        sample_metrics_df: Sample metrics DataFrame.
    """
    from frontend.components.metrics_chart import plot_accuracy_curves

    plot_accuracy_curves(sample_metrics_df)

    # Verify plotly chart was called
    assert mock_st_plotly_chart.called

    # Get the figure
    fig = mock_st_plotly_chart.call_args[0][0]

    # Should only have accuracy metric (1 trace)
    assert len(fig.data) == 1
    assert fig.data[0].name == "accuracy"

    # Verify title
    assert fig.layout.title.text == "Accuracy Metrics"


def test_plot_accuracy_curves_no_data(mock_st_info: MockerFixture) -> None:
    """Test accuracy curves with no accuracy metrics shows info.

    Args:
        mock_st_info: Mocked info function.
    """
    from frontend.components.metrics_chart import plot_accuracy_curves

    # DataFrame with no accuracy metrics
    df = pl.DataFrame(
        {
            "step": [0, 1, 2],
            "metric": ["loss", "loss", "loss"],
            "value": [1.0, 0.8, 0.5],
            "timestamp": [1000, 1001, 1002],
        }
    )

    plot_accuracy_curves(df)

    # Should show info message
    assert mock_st_info.called
    assert "No accuracy metrics available" in mock_st_info.call_args[0][0]
