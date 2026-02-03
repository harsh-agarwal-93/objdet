"""Streamlit session state management."""

from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    """Initialize Streamlit session state with defaults."""
    # Active jobs monitoring
    if "active_jobs" not in st.session_state:
        st.session_state.active_jobs = {}

    # MLFlow cache
    if "experiments" not in st.session_state:
        st.session_state.experiments = None

    if "runs_cache" not in st.session_state:
        st.session_state.runs_cache = {}

    # Page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    # Selected items
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None


def add_active_job(task_id: str, job_info: dict) -> None:
    """Add a job to active jobs monitoring.

    Args:
        task_id: Celery task ID.
        job_info: Job information dictionary.
    """
    if "active_jobs" not in st.session_state:
        st.session_state.active_jobs = {}

    st.session_state.active_jobs[task_id] = job_info


def remove_active_job(task_id: str) -> None:
    """Remove a job from active jobs monitoring.

    Args:
        task_id: Celery task ID.
    """
    if "active_jobs" in st.session_state and task_id in st.session_state.active_jobs:
        del st.session_state.active_jobs[task_id]


def get_active_jobs() -> dict:
    """Get all active jobs.

    Returns:
        Dictionary of active jobs.
    """
    return st.session_state.get("active_jobs", {})
