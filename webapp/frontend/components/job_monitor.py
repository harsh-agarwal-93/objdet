"""Job monitoring component."""

from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from frontend.api.client import get_client
from frontend.components.status_badge import status_badge
from frontend.utils.session import get_active_jobs, remove_active_job


def job_monitor() -> None:
    """Display active jobs monitor with auto-refresh."""
    active_jobs = get_active_jobs()

    if not active_jobs:
        st.info("No active jobs")
        return

    # Auto-refresh every 10 seconds
    st_autorefresh(interval=10000, key="job_monitor_refresh")

    st.subheader("Active Jobs")

    client = get_client()

    for task_id, job_info in list(active_jobs.items()):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**{job_info.get('name', 'Training Job')}**")
                st.caption(f"Task ID: `{task_id[:16]}...`")

            # Get current status
            try:
                status_data = client.get_task_status(task_id)
                current_status = status_data.get("status", "UNKNOWN")

                with col2:
                    status_badge(current_status)

                with col3:
                    if current_status in ["PENDING", "STARTED"] and st.button(
                        "Cancel", key=f"cancel_{task_id}"
                    ):
                        client.cancel_task(task_id)
                        remove_active_job(task_id)
                        st.rerun()

                # Show progress if available
                if current_status == "STARTED" and "progress" in status_data:
                    progress = status_data["progress"]
                    if isinstance(progress, dict):
                        current_epoch = progress.get("current_epoch", 0)
                        total_epochs = progress.get("total_epochs", 100)
                        progress_pct = current_epoch / total_epochs if total_epochs > 0 else 0
                        st.progress(progress_pct, text=f"Epoch {current_epoch}/{total_epochs}")

                # Remove completed/failed jobs from monitoring
                if current_status in ["SUCCESS", "FAILURE", "REVOKED"]:
                    remove_active_job(task_id)
                    st.rerun()

            except Exception as e:
                st.error(f"Error fetching status: {e!s}")

            st.divider()
