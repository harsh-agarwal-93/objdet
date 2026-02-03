"""Models page - Training and model management (Workflow 1)."""

from __future__ import annotations

import streamlit as st
from frontend.api.client import get_client
from frontend.components.job_monitor import job_monitor
from frontend.components.metrics_chart import plot_loss_curves
from frontend.components.status_badge import status_badge
from frontend.utils.formatting import format_timestamp
from frontend.utils.session import add_active_job, init_session_state

st.set_page_config(page_title="Models", page_icon="🧠", layout="wide")

init_session_state()

st.title("🧠 Model Training")

client = get_client()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Previous Runs", "🚀 New Training", "⚡ Active Jobs"])

# === Previous Runs Tab ===
with tab1:
    st.subheader("Training Runs from MLFlow")

    try:
        # List runs
        runs = client.list_runs(max_results=50)

        if not runs:
            st.info("No training runs found in MLFlow")
        else:
            # Display runs table
            for run in runs:
                with st.expander(f"**{run['run_name']}** - {run['status']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.write("**Run ID:**", f"`{run['run_id']}`")
                        st.write("**Experiment ID:**", run["experiment_id"])

                    with col2:
                        status_badge(run["status"])

                    with col3:
                        if run["start_time"]:
                            st.caption(f"Started: {format_timestamp(run['start_time'])}")

                    # Show details button
                    if st.button("View Details", key=f"details_{run['run_id']}"):
                        st.session_state.selected_run_id = run["run_id"]

                    # If selected, show metrics
                    if st.session_state.get("selected_run_id") == run["run_id"]:
                        st.divider()

                        # Get run details
                        details = client.get_run_details(run["run_id"])

                        # Show metrics
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.write("**Parameters:**")
                            if details.get("params"):
                                for param, value in details["params"].items():
                                    st.caption(f"{param}: {value}")
                            else:
                                st.caption("No parameters logged")

                        with col_b:
                            st.write("**Final Metrics:**")
                            if details.get("metrics"):
                                for metric, value in details["metrics"].items():
                                    st.caption(f"{metric}: {value:.4f}")
                            else:
                                st.caption("No metrics logged")

                        # Plot metrics
                        try:
                            metrics_df = client.get_run_metrics(run["run_id"])
                            if not metrics_df.is_empty():
                                st.write("**Training Curves:**")
                                plot_loss_curves(metrics_df)
                            else:
                                st.info("No metric history available")
                        except Exception as e:
                            st.warning(f"Could not load metrics: {e!s}")

    except Exception as e:
        st.error(f"Error loading runs: {e!s}")

# === New Training Tab ===
with tab2:
    st.subheader("Submit New Training Job")

    with st.form("training_form"):
        col1, col2 = st.columns(2)

        with col1:
            run_name = st.text_input("Run Name", value="YOLOv8 Training Run")
            model_architecture = st.selectbox(
                "Model Architecture",
                ["yolov8", "yolov11", "retinanet", "fcos"],
            )
            dataset = st.text_input("Dataset", value="coco2017")
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=32)

        with col2:
            learning_rate = st.number_input(
                "Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f"
            )
            optimizer = st.selectbox("Optimizer", ["adam", "sgd", "adamw"])
            gpu = st.selectbox("GPU Selection", ["auto", "gpu0", "gpu1", "multi"])
            priority = st.selectbox("Priority", ["low", "normal", "high"])
            mixed_precision = st.selectbox("Mixed Precision", ["fp16", "fp32", "bf16"])

        # Checkboxes
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            save_checkpoints = st.checkbox("Save Checkpoints", value=True)
            early_stopping = st.checkbox("Early Stopping", value=True)
        with col_b:
            log_to_mlflow = st.checkbox("Log to MLFlow", value=True)
        with col_c:
            data_augmentation = st.checkbox("Data Augmentation", value=True)

        # Submit button
        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)

        if submitted:
            # Create training config
            config = {
                "name": run_name,
                "model_architecture": model_architecture,
                "dataset": dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "gpu": gpu,
                "priority": priority,
                "mixed_precision": mixed_precision,
                "save_checkpoints": save_checkpoints,
                "early_stopping": early_stopping,
                "log_to_mlflow": log_to_mlflow,
                "data_augmentation": data_augmentation,
            }

            try:
                # Submit job
                response = client.submit_training_job(config)

                st.success(f"Training job submitted! Task ID: `{response['task_id']}`")

                # Add to active jobs monitoring
                add_active_job(
                    response["task_id"],
                    {
                        "name": run_name,
                        "created_at": response["created_at"],
                    },
                )

                # Switch to active jobs tab
                st.info("Check the 'Active Jobs' tab to monitor progress")

            except Exception as e:
                st.error(f"Failed to submit training job: {e!s}")

# === Active Jobs Tab ===
with tab3:
    job_monitor()
