"""Tests for status badge component."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_st_markdown(mocker: MockerFixture) -> MockerFixture:
    """Mock streamlit.markdown.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Mocked markdown function.
    """
    return mocker.patch("frontend.components.status_badge.st.markdown")


def test_status_badge_queued(mock_st_markdown: MockerFixture) -> None:
    """Test queued status displays with purple color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("queued")

    # Verify markdown was called
    assert mock_st_markdown.called
    call_args = mock_st_markdown.call_args[0][0]

    # Check purple color and queued icon
    assert "#9333ea" in call_args
    assert "🟣" in call_args
    assert "Queued" in call_args


def test_status_badge_pending(mock_st_markdown: MockerFixture) -> None:
    """Test pending status displays with purple color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("PENDING")  # Test case insensitivity

    call_args = mock_st_markdown.call_args[0][0]
    assert "#9333ea" in call_args
    assert "🟣" in call_args
    assert "Pending" in call_args


def test_status_badge_running(mock_st_markdown: MockerFixture) -> None:
    """Test running status displays with blue color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("running")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#3b82f6" in call_args
    assert "🔵" in call_args
    assert "Running" in call_args


def test_status_badge_started(mock_st_markdown: MockerFixture) -> None:
    """Test started status displays with blue color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("STARTED")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#3b82f6" in call_args
    assert "🔵" in call_args
    assert "Started" in call_args


def test_status_badge_completed(mock_st_markdown: MockerFixture) -> None:
    """Test completed status displays with green color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("completed")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#10b981" in call_args
    assert "🟢" in call_args
    assert "Completed" in call_args


def test_status_badge_success(mock_st_markdown: MockerFixture) -> None:
    """Test success status displays with green color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("SUCCESS")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#10b981" in call_args
    assert "🟢" in call_args
    assert "Success" in call_args


def test_status_badge_failed(mock_st_markdown: MockerFixture) -> None:
    """Test failed status displays with red color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("failed")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#ef4444" in call_args
    assert "🔴" in call_args
    assert "Failed" in call_args


def test_status_badge_failure(mock_st_markdown: MockerFixture) -> None:
    """Test failure status displays with red color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("FAILURE")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#ef4444" in call_args
    assert "🔴" in call_args
    assert "Failure" in call_args


def test_status_badge_cancelled(mock_st_markdown: MockerFixture) -> None:
    """Test cancelled status displays with gray color.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("cancelled")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#6b7280" in call_args
    assert "⚪" in call_args
    assert "Cancelled" in call_args


def test_status_badge_unknown(mock_st_markdown: MockerFixture) -> None:
    """Test unknown status displays with default fallback.

    Args:
        mock_st_markdown: Mocked streamlit markdown.
    """
    from frontend.components.status_badge import status_badge

    status_badge("UNKNOWN_STATUS")

    call_args = mock_st_markdown.call_args[0][0]
    assert "#6b7280" in call_args  # Default gray
    assert "⚫" in call_args  # Default icon
    assert "UNKNOWN_STATUS" in call_args  # Original status
