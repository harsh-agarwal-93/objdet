"""Effects page - Effect training and validation (Workflow 2 placeholder)."""

from __future__ import annotations

import streamlit as st
from frontend.utils.session import init_session_state

st.set_page_config(page_title="Effects", page_icon="⚡", layout="wide")

init_session_state()

st.title("⚡ Effects Training")

st.info(
    """
    **Note:** Effect training integration is coming soon. This page
    demonstrates the planned UI structure.
    """
)

# Tabs
tab1, tab2, tab3 = st.tabs(["📚 Effect Library", "✨ Train New Effect", "🧪 Validation Runs"])

with tab1:
    st.subheader("Effect Library")

    st.markdown(
        """
        This section will display:
        - Grid view of trained effects
        - Category filters (Basic, Polygon, Complex)
        - Effect cards showing success rate and transparency
        - Validation run button per effect
        """
    )

    # Placeholder cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Effects", "24")

    with col2:
        st.metric("Validated", "18")

    with col3:
        st.metric("Avg Success Rate", "92.5%")

with tab2:
    st.subheader("Train New Effect")

    st.markdown(
        """
        **Planned Features:**
        - Effect configuration form
        - Training parameters (epochs, batch size)
        - Dataset selection
        - Submit to Celery for training
        """
    )

    with st.form("effect_form"):
        effect_name = st.text_input("Effect Name")
        effect_type = st.selectbox("Effect Type", ["Basic", "Polygon", "Complex"])
        grid_size = st.slider("Grid Size", 8, 64, 16)

        submitted = st.form_submit_button("Train Effect (Coming Soon)", disabled=True)

        if submitted:
            st.warning("Effect training integration not yet implemented")

with tab3:
    st.subheader("Validation Runs")

    st.markdown(
        """
        **Planned Features:**
        - Pull validation artifact data from MLFlow
        - Display validation results
        - Show success rate and quality metrics
        """
    )

    st.info("No validation runs available. Effect training system coming soon.")
