"""Loadset page - Deployment packaging (Placeholder)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Loadset", page_icon="📁", layout="wide")

st.title("📁 Loadset Builder")

st.info(
    """
    **Coming Soon:** Deployment packaging system for models, effects, and configurations.
    """
)

st.markdown(
    """
    ## Planned Features

    **Package Management:**
    - Create deployment packages (loadsets)
    - Version control for loadsets
    - Package metadata and documentation

    **Component Selection:**
    - Select trained models from MLFlow
    - Include effects and scene templates
    - Add configuration files
    - Bundle dependencies

    **Export Options:**
    - Package formats (Docker, ONNX, TorchScript)
    - Target platforms (edge devices, cloud)
    - Optimization settings

    **Deployment:**
    - Push to container registry
    - Deploy to staging/production
    - Rollback management
    """
)

st.divider()

# Placeholder form
st.subheader("Create New Loadset (Preview)")

with st.form("loadset_form"):
    loadset_name = st.text_input("Loadset Name", value="production-v1.0")

    col1, col2 = st.columns(2)

    with col1:
        model_select = st.multiselect("Select Models", ["YOLOv8-v4", "RetinaNet-v2", "FCOS-v3"])
        effects_select = st.multiselect("Select Effects", ["Basic-Polygon", "Complex-Blur"])

    with col2:
        target_platform = st.selectbox("Target Platform", ["Docker", "Edge Device", "Cloud"])
        optimization = st.selectbox("Optimization", ["Speed", "Accuracy", "Balanced"])

    submitted = st.form_submit_button("Build Loadset (Coming Soon)", disabled=True)

    if submitted:
        st.warning("Loadset builder not yet implemented")
