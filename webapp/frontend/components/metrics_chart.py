"""Metrics chart component."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st


def plot_metrics_chart(df: pl.DataFrame, title: str = "Training Metrics") -> None:
    """Plot training metrics with Plotly.

    Args:
        df: Polars DataFrame with columns: step, metric, value.
        title: Chart title.
    """
    if df.is_empty():
        st.info("No metrics data available")
        return

    fig = go.Figure()

    # Get unique metrics
    unique_metrics = df["metric"].unique().to_list()

    # Add trace for each metric
    for metric in unique_metrics:
        metric_data = df.filter(pl.col("metric") == metric)

        fig.add_trace(
            go.Scatter(
                x=metric_data["step"].to_list(),
                y=metric_data["value"].to_list(),
                mode="lines+markers",
                name=metric,
                line={"width": 2},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Step/Epoch",
        yaxis_title="Value",
        template="plotly_dark",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_loss_curves(df: pl.DataFrame) -> None:
    """Plot train/val loss curves.

    Args:
        df: Polars DataFrame with train/val loss metrics.
    """
    # Filter for loss metrics
    loss_df = df.filter(
        (pl.col("metric").str.contains("loss")) | (pl.col("metric").str.contains("Loss"))
    )

    if loss_df.is_empty():
        st.info("No loss metrics available")
        return

    plot_metrics_chart(loss_df, title="Loss Curves")


def plot_accuracy_curves(df: pl.DataFrame) -> None:
    """Plot accuracy curves.

    Args:
        df: Polars DataFrame with accuracy metrics.
    """
    # Filter for accuracy/precision/recall metrics
    acc_df = df.filter(
        (pl.col("metric").str.contains("acc"))
        | (pl.col("metric").str.contains("precision"))
        | (pl.col("metric").str.contains("recall"))
        | (pl.col("metric").str.contains("f1"))
    )

    if acc_df.is_empty():
        st.info("No accuracy metrics available")
        return

    plot_metrics_chart(acc_df, title="Accuracy Metrics")
