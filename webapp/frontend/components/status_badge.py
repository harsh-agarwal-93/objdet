"""Status badge component."""

from __future__ import annotations

import streamlit as st


def status_badge(status: str) -> None:
    """Display a colored status badge.

    Args:
        status: Status string (queued, running, completed, failed, pending, started, success, failure).
    """
    status_lower = status.lower()

    # Map status to colors and icons
    status_map = {
        "queued": ("🟣", "#9333ea", "Queued"),
        "pending": ("🟣", "#9333ea", "Pending"),
        "running": ("🔵", "#3b82f6", "Running"),
        "started": ("🔵", "#3b82f6", "Started"),
        "completed": ("🟢", "#10b981", "Completed"),
        "success": ("🟢", "#10b981", "Success"),
        "finished": ("🟢", "#10b981", "Finished"),
        "failed": ("🔴", "#ef4444", "Failed"),
        "failure": ("🔴", "#ef4444", "Failure"),
        "cancelled": ("⚪", "#6b7280", "Cancelled"),
        "revoked": ("⚪", "#6b7280", "Revoked"),
    }

    icon, color, display_text = status_map.get(status_lower, ("⚫", "#6b7280", status))

    st.markdown(
        f"""
        <span style="
            background-color: {color}22;
            color: {color};
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
        ">
            {icon} {display_text}
        </span>
        """,
        unsafe_allow_html=True,
    )
