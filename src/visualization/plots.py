"""
Reusable chart functions for forecasting visualisation (Plotly + Matplotlib).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


# ---------------------------------------------------------------------------
# Plotly — interactive
# ---------------------------------------------------------------------------

def forecast_ribbon_plotly(
    history: pd.Series,
    forecast: pd.DataFrame,
    title: str = "Electricity Demand Forecast",
    history_days: int = 14,
) -> go.Figure:
    """Forecast + confidence ribbon chart."""
    hist = history.iloc[-history_days * 24 :]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Historical",
        line=dict(color="#1f77b4", width=1.5),
    ))

    if "lower_95" in forecast.columns and "upper_95" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast.index.to_series(), forecast.index.to_series()[::-1]]),
            y=pd.concat([forecast["upper_95"], forecast["lower_95"][::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

    if "lower_80" in forecast.columns and "upper_80" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast.index.to_series(), forecast.index.to_series()[::-1]]),
            y=pd.concat([forecast["upper_80"], forecast["lower_80"][::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI",
        ))

    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["forecast"],
        mode="lines", name="Forecast",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def model_comparison_plotly(results: pd.DataFrame) -> go.Figure:
    """Bar chart comparing model MAE / RMSE / MAPE."""
    metrics = ["MAE", "RMSE", "MAPE"]
    fig = go.Figure()
    for metric in metrics:
        if metric in results.columns:
            fig.add_trace(go.Bar(name=metric, x=results["model"], y=results[metric]))
    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        xaxis_title="Model",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib — static / publication quality
# ---------------------------------------------------------------------------

def plot_residuals(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Model") -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(residuals.index, residuals.values, alpha=0.6, linewidth=0.8)
    axes[0].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[0].set_title(f"{model_name} — Residuals over time")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual (MW)")

    sns.histplot(residuals, kde=True, ax=axes[1], bins=50)
    axes[1].set_title(f"{model_name} — Residual distribution")
    axes[1].set_xlabel("Residual (MW)")

    plt.tight_layout()


def plot_forecast_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast vs Actual",
) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(y_true.index, y_true.values, label="Actual", alpha=0.8)
    plt.plot(y_pred.index, y_pred.values, label="Forecast", linestyle="--", alpha=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()


def plot_hourly_profile(series: pd.Series) -> None:
    profile = series.groupby(series.index.hour).mean()
    plt.figure(figsize=(10, 4))
    plt.bar(profile.index, profile.values, color="#1f77b4", alpha=0.8)
    plt.title("Average Hourly Load Profile")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Load (MW)")
    plt.xticks(range(24))
    plt.tight_layout()
