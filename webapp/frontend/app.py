"""ObjDet Web Application - Main Streamlit App."""

from __future__ import annotations

import streamlit as st

from frontend.utils.session import init_session_state

# Page configuration
st.set_page_config(
    page_title="ObjDet Platform",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "About": "ObjDet ML Platform - Object Detection Training and Management",
    },
)

# Initialize session state
init_session_state()

# Custom CSS for dark theme enhancements
st.markdown(
    """
    <style>
    /* Card styling */
    .stCard {
        background-color: #1a1f29;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Button styling */
    .stButton>button {
        border-radius: 0.375rem;
        font-weight: 500;
    }

    /* Metric card */
    div[data-testid="metric-container"] {
        background-color: #1a1f29;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🧠 ObjDet ML Platform")
st.markdown("**Object Detection Training and Management**")

st.divider()

# Welcome message
st.markdown(
    """
    ### Welcome to the ObjDet Platform

    This platform provides a comprehensive interface for:
    - **Training** object detection models with MLFlow tracking
    - **Managing** model artifacts and experiments
    - **Generating** synthetic training data
    - **Deploying** models for production

    👈 **Use the sidebar to navigate** between different workflows.
    """
)

st.divider()

# System status
st.subheader("System Status")

try:
    from frontend.api.client import get_client

    client = get_client()
    status = client.get_system_status()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Celery Status",
            status["services"]["celery"]["status"].upper(),
            delta="Connected"
            if status["services"]["celery"]["status"] == "connected"
            else "Disconnected",
        )

    with col2:
        st.metric(
            "MLFlow Status",
            status["services"]["mlflow"]["status"].upper(),
            delta="Connected"
            if status["services"]["mlflow"]["status"] == "connected"
            else "Disconnected",
        )

except Exception as e:
    st.error(f"Failed to connect to backend: {e!s}")
    st.info("Make sure the FastAPI backend is running at the configured `BACKEND_URL`")

st.divider()

# Quick stats
st.subheader("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/2_🧠_Models.py", label="🧠 Train Models", use_container_width=True)

with col2:
    st.page_link("pages/3_⚡_Effects.py", label="⚡ Manage Effects", use_container_width=True)

with col3:
    st.page_link("pages/4_📦_Synthetic_Data.py", label="📦 Generate Data", use_container_width=True)
