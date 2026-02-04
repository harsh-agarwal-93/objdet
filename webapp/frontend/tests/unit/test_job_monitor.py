"""Tests for job monitor component."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_get_client(mocker: MockerFixture) -> MagicMock:
    """Mock get_client function.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Mocked client.
    """
    mock_client = MagicMock()
    mock_client.get_task_status.return_value = {
        "task_id": "test-123",
        "status": "STARTED",
        "result": None,
        "error": None,
    }
    mocker.patch("frontend.components.job_monitor.get_client", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_session_utils(mocker: MockerFixture) -> dict[str, MagicMock]:
    """Mock session utility functions.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Dictionary of mocked session functions.
    """
    return {
        "get_active_jobs": mocker.patch("frontend.components.job_monitor.get_active_jobs"),
        "remove_active_job": mocker.patch("frontend.components.job_monitor.remove_active_job"),
    }


@pytest.fixture
def mock_streamlit_widgets(mocker: MockerFixture) -> dict[str, MagicMock]:
    """Mock Streamlit widgets for job monitor.

    Args:
        mocker: Pytest mocker fixture.

    Returns:
        Dictionary of mocked Streamlit functions.
    """
    # Mock container as context manager
    mock_container = MagicMock()
    mock_container.__enter__ = MagicMock(return_value=mock_container)
    mock_container.__exit__ = MagicMock(return_value=None)

    # Mock columns to return list of column contexts
    mock_col1 = MagicMock()
    mock_col1.__enter__ = MagicMock(return_value=mock_col1)
    mock_col1.__exit__ = MagicMock(return_value=None)

    mock_col2 = MagicMock()
    mock_col2.__enter__ = MagicMock(return_value=mock_col2)
    mock_col2.__exit__ = MagicMock(return_value=None)

    mock_col3 = MagicMock()
    mock_col3.__enter__ = MagicMock(return_value=mock_col3)
    mock_col3.__exit__ = MagicMock(return_value=None)

    return {
        "st_autorefresh": mocker.patch("frontend.components.job_monitor.st_autorefresh"),
        "info": mocker.patch("frontend.components.job_monitor.st.info"),
        "subheader": mocker.patch("frontend.components.job_monitor.st.subheader"),
        "container": mocker.patch(
            "frontend.components.job_monitor.st.container", return_value=mock_container
        ),
        "columns": mocker.patch(
            "frontend.components.job_monitor.st.columns",
            return_value=[mock_col1, mock_col2, mock_col3],
        ),
        "write": mocker.patch("frontend.components.job_monitor.st.write"),
        "caption": mocker.patch("frontend.components.job_monitor.st.caption"),
        "button": mocker.patch("frontend.components.job_monitor.st.button", return_value=False),
        "progress": mocker.patch("frontend.components.job_monitor.st.progress"),
        "rerun": mocker.patch("frontend.components.job_monitor.st.rerun"),
        "divider": mocker.patch("frontend.components.job_monitor.st.divider"),
        "error": mocker.patch("frontend.components.job_monitor.st.error"),
    }


def test_job_monitor_no_jobs(
    mock_session_utils: dict[str, MagicMock], mock_streamlit_widgets: dict[str, MagicMock]
) -> None:
    """Test job monitor shows info when no active jobs.

    Args:
        mock_session_utils: Mocked session utilities.
        mock_streamlit_widgets: Mocked Streamlit widgets.
    """
    from frontend.components.job_monitor import job_monitor

    # No active jobs
    mock_session_utils["get_active_jobs"].return_value = {}

    job_monitor()

    # Should show info message
    assert mock_streamlit_widgets["info"].called
    assert "No active jobs" in mock_streamlit_widgets["info"].call_args[0][0]


def test_job_monitor_display_jobs(
    mock_get_client: MagicMock,
    mock_session_utils: dict[str, MagicMock],
    mock_streamlit_widgets: dict[str, MagicMock],
) -> None:
    """Test job monitor displays active jobs.

    Args:
        mock_get_client: Mocked backend client.
        mock_session_utils: Mocked session utilities.
        mock_streamlit_widgets: Mocked Streamlit widgets.
    """
    from frontend.components.job_monitor import job_monitor

    # Set up active jobs
    mock_session_utils["get_active_jobs"].return_value = {
        "task-123": {"name": "YOLOv8 Training", "submitted_at": 1234567890}
    }

    mock_get_client.get_task_status.return_value = {
        "task_id": "task-123",
        "status": "STARTED",
        "result": None,
        "error": None,
    }

    job_monitor()

    # Should show subheader
    assert mock_streamlit_widgets["subheader"].called

    # Should create container
    assert mock_streamlit_widgets["container"].called

    # Should create columns
    assert mock_streamlit_widgets["columns"].called

    # Should fetch task status
    assert mock_get_client.get_task_status.called
    assert mock_get_client.get_task_status.call_args[0][0] == "task-123"


def test_job_monitor_cancel_button(
    mock_get_client: MagicMock,
    mock_session_utils: dict[str, MagicMock],
    mock_streamlit_widgets: dict[str, MagicMock],
) -> None:
    """Test cancel button interaction.

    Args:
        mock_get_client: Mocked backend client.
        mock_session_utils: Mocked session utilities.
        mock_streamlit_widgets: Mocked Streamlit widgets.
    """
    from frontend.components.job_monitor import job_monitor

    # Set up active jobs
    mock_session_utils["get_active_jobs"].return_value = {"task-123": {"name": "YOLOv8 Training"}}

    mock_get_client.get_task_status.return_value = {
        "task_id": "task-123",
        "status": "STARTED",
        "result": None,
        "error": None,
    }

    # Simulate button click
    mock_streamlit_widgets["button"].return_value = True

    job_monitor()

    # Should call cancel_task
    assert mock_get_client.cancel_task.called
    assert mock_get_client.cancel_task.call_args[0][0] == "task-123"

    # Should remove job from active list
    assert mock_session_utils["remove_active_job"].called
    assert mock_session_utils["remove_active_job"].call_args[0][0] == "task-123"

    # Should trigger rerun
    assert mock_streamlit_widgets["rerun"].called


def test_job_monitor_auto_cleanup(
    mock_get_client: MagicMock,
    mock_session_utils: dict[str, MagicMock],
    mock_streamlit_widgets: dict[str, MagicMock],
) -> None:
    """Test completed jobs are automatically removed.

    Args:
        mock_get_client: Mocked backend client.
        mock_session_utils: Mocked session utilities.
        mock_streamlit_widgets: Mocked Streamlit widgets.
    """
    from frontend.components.job_monitor import job_monitor

    # Set up active jobs
    mock_session_utils["get_active_jobs"].return_value = {"task-123": {"name": "YOLOv8 Training"}}

    # Job is completed
    mock_get_client.get_task_status.return_value = {
        "task_id": "task-123",
        "status": "SUCCESS",
        "result": None,
        "error": None,
    }

    job_monitor()

    # Should remove completed job
    assert mock_session_utils["remove_active_job"].called
    assert mock_session_utils["remove_active_job"].call_args[0][0] == "task-123"

    # Should trigger rerun
    assert mock_streamlit_widgets["rerun"].called
