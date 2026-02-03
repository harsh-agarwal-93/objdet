"""Synthetic Data page - Data generation (Workflow 3 placeholder)."""

from __future__ import annotations

import streamlit as st
from frontend.utils.session import init_session_state

st.set_page_config(page_title="Synthetic Data", page_icon="📦", layout="wide")

init_session_state()

st.title("📦 Synthetic Data Generation")

st.info(
    """
    **Note:** Synthetic data generation integration is coming soon.
    This page demonstrates the planned UI structure.
    """
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎨 CAD Models", "⚙️ Generation Config", "📊 Previous Jobs"])

with tab1:
    st.subheader("CAD Model Database")

    st.markdown(
        """
        **Planned Features:**
        - Browse CAD models by category
        - Upload new CAD models
        - Model preview and metadata
        """
    )

    # Placeholder metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total CAD Models", "147")

    with col2:
        st.metric("Categories", "8")

    with col3:
        st.metric("Recently Added", "12")

    st.button("Upload CAD Model (Coming Soon)", disabled=True, use_container_width=True)

with tab2:
    st.subheader("Data Generation Configuration")

    st.markdown("**Camera Angles:**")
    col1, col2 = st.columns(2)

    with col1:
        azimuth = st.slider("Azimuth", 0, 360, 180)

    with col2:
        elevation = st.slider("Elevation", -90, 90, 45)

    st.markdown("**Simulation Parameters:**")
    col3, col4 = st.columns(2)

    with col3:
        range_bins = st.number_input("Range Bins", 100, 2000, 512)

    with col4:
        crossrange_bins = st.number_input("Crossrange Bins", 100, 2000, 512)

    st.markdown("**Render Options:**")
    col5, col6 = st.columns(2)

    with col5:
        seg_masks = st.checkbox("Segmentation Masks", value=True)
        bg_noise = st.checkbox("Background Noise", value=True)

    with col6:
        json_export = st.checkbox("Export JSON", value=True)
        pixel_res = st.number_input("Pixel Resolution", 256, 2048, 640)

    st.button("🚀 Generate Data (Coming Soon)", disabled=True, use_container_width=True)

with tab3:
    st.subheader("Previous Generation Jobs")

    st.markdown(
        """
        **Planned Features:**
        - List of completed generation jobs
        - Pull artifacts from MLFlow (datasets tagged with `synthetic_data=true`)
        - Download generated datasets
        - View generation statistics
        """
    )

    st.info("No synthetic data jobs available. Data generation system coming soon.")
