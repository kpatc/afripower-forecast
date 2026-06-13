"""
Streamlit dashboard — AfriPower Forecast.
Run: make dashboard  (or: venv/bin/streamlit run dashboard/app.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="AfriPower Forecast",
    page_icon="⚡",
    layout="wide",
)

CITIES = ["tetouan", "laayoune", "boujdour", "foum_eloued", "marrakech"]
CITY_LABELS = {
    "tetouan":    "Tétouan",
    "laayoune":   "Laâyoune",
    "boujdour":   "Boujdour",
    "foum_eloued":"Foum El Oued",
    "marrakech":  "Marrakech",
}
METRICS = {
    "SARIMA":   {"scope": "Tétouan only", "MAE": 0.370, "RMSE": 0.437, "MAPE": 96.9,  "Coverage80": None},
    "Prophet":  {"scope": "Tétouan only", "MAE": 0.288, "RMSE": 0.368, "MAPE": 103.3, "Coverage80": None},
    "XGBoost":  {"scope": "All 5 cities", "MAE": 0.124, "RMSE": 0.225, "MAPE": 58.6,  "Coverage80": 76.6},
    "LightGBM": {"scope": "All 5 cities", "MAE": 0.118, "RMSE": 0.204, "MAPE": 56.0,  "Coverage80": 73.9},
    "LSTM":     {"scope": "All 5 cities", "MAE": 0.543, "RMSE": 0.686, "MAPE": 129.2, "Coverage80": None},
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ AfriPower Forecast")
st.sidebar.caption("Moroccan electricity demand intelligence")
st.sidebar.markdown("---")

city_label = st.sidebar.selectbox("City", list(CITY_LABELS.values()), index=0)
city = [k for k, v in CITY_LABELS.items() if v == city_label][0]

model_choice = st.sidebar.selectbox("Model", ["LightGBM", "XGBoost"], index=0)
horizon_days = st.sidebar.slider("Forecast horizon (days)", 1, 7, 7)
show_ci = st.sidebar.checkbox("Show 80% prediction interval", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sources**")
st.sidebar.markdown("- Tétouan 2017 (Kaggle)\n- UCI Smart Meters 2022–2024\n- Open-Meteo weather")
st.sidebar.markdown("**Models**")
st.sidebar.markdown("SARIMA · Prophet · XGBoost · LightGBM · LSTM")


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame | None:
    p = PROCESSED / "features.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=3600)
def load_norm_stats() -> pd.DataFrame | None:
    p = PROCESSED / "norm_stats.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


@st.cache_resource
def load_gbm_model(name: str):
    p = PROCESSED / f"{name.lower()}.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


def city_test_split(df: pd.DataFrame, city: str, test_ratio: float = 0.15):
    mask = df[f"city_{city}"] == 1
    city_df = df[mask].sort_index()
    n = len(city_df)
    split = int(n * (1 - test_ratio))
    return city_df.iloc[:split], city_df.iloc[split:]


def denorm(values: np.ndarray, city: str, norm_stats: pd.DataFrame) -> np.ndarray:
    mean = norm_stats.loc[city, "mean_kW"]
    std  = norm_stats.loc[city, "std_kW"]
    return values * std + mean


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_forecast, tab_eda, tab_models, tab_feat = st.tabs(
    ["📈 Forecast", "🔍 EDA", "🏆 Model Comparison", "🔬 Feature Importance"]
)


# ── Tab 1: Forecast ─────────────────────────────────────────────────────────
with tab_forecast:
    st.subheader(f"J+1 → J+{horizon_days} Forecast — {city_label} ({model_choice})")

    df = load_features()
    norm_stats = load_norm_stats()

    if df is None:
        st.warning("No feature data found. Run `make train` first.")
    else:
        models = load_gbm_model(model_choice)

        if models is None:
            st.warning(f"No trained {model_choice} model found in `data/processed/`. Run `make train`.")
        else:
            from src.models.ml_models import _split

            _, test_city = city_test_split(df, city)
            horizon_h = horizon_days * 24
            test_slice = test_city.iloc[:horizon_h]

            X_test, y_test = _split(test_slice)
            preds = pd.DataFrame({
                "forecast":  models[0.5].predict(X_test),
                "lower_80":  models[0.1].predict(X_test),
                "upper_80":  models[0.9].predict(X_test),
            }, index=X_test.index)

            # Denormalise to kW if norm_stats available
            if norm_stats is not None and city in norm_stats.index:
                y_actual = denorm(y_test.values, city, norm_stats)
                y_fc     = denorm(preds["forecast"].values, city, norm_stats)
                y_lo     = denorm(preds["lower_80"].values, city, norm_stats)
                y_hi     = denorm(preds["upper_80"].values, city, norm_stats)
                unit = "kW"
            else:
                y_actual = y_test.values
                y_fc     = preds["forecast"].values
                y_lo     = preds["lower_80"].values
                y_hi     = preds["upper_80"].values
                unit = "z-score"

            idx = preds.index

            fig = go.Figure()
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=list(idx) + list(idx[::-1]),
                    y=list(y_hi) + list(y_lo[::-1]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% PI", showlegend=True,
                ))
            fig.add_trace(go.Scatter(x=idx, y=y_actual, mode="lines",
                                     line=dict(color="#636EFA", width=1.5),
                                     name=f"Actual ({unit})"))
            fig.add_trace(go.Scatter(x=idx, y=y_fc, mode="lines",
                                     line=dict(color="#EF553B", width=2, dash="dash"),
                                     name=f"Forecast ({unit})"))
            fig.update_layout(
                height=420, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Time", yaxis_title=unit,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_column_width=True)

            col1, col2, col3, col4 = st.columns(4)
            mae = np.mean(np.abs(y_actual - y_fc))
            rmse = np.sqrt(np.mean((y_actual - y_fc) ** 2))
            coverage = np.mean((y_actual >= y_lo) & (y_actual <= y_hi)) * 100
            col1.metric("Peak Forecast", f"{y_fc.max():,.0f} {unit}")
            col2.metric("MAE", f"{mae:.3f}")
            col3.metric("RMSE", f"{rmse:.3f}")
            col4.metric("Coverage 80%", f"{coverage:.1f}%")


# ── Tab 2: EDA ───────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    eda_plots = [
        ("eda_01_data_coverage.png",    "Dataset Coverage by City"),
        ("eda_02_hourly_profile.png",   "Hourly Load Profile"),
        ("eda_03_weekly_profile.png",   "Weekly Load Profile"),
        ("eda_04_monthly_seasonality.png", "Monthly Seasonality"),
        ("eda_05_distributions.png",    "Load Distributions"),
        ("eda_06_timeseries_zoom.png",  "Time-series Zoom"),
        ("eda_07_correlation_matrix.png","Correlation Matrix"),
        ("eda_08_temp_vs_load.png",     "Temperature vs Load"),
        ("eda_09_heatmap_dow_hour.png", "Day-of-Week × Hour Heatmap"),
    ]

    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(eda_plots):
                fname, title = eda_plots[i + j]
                p = ASSETS / fname
                if p.exists():
                    col.markdown(f"**{title}**")
                    col.image(str(p), use_column_width=True)


# ── Tab 3: Model Comparison ──────────────────────────────────────────────────
with tab_models:
    st.subheader("Model Benchmark — Hold-out Test Set (last 15%)")
    st.caption("MAE/RMSE in z-score units. MAPE inflated by near-zero z-score values.")

    results_df = pd.DataFrame(METRICS).T.reset_index().rename(columns={"index": "Model"})

    # Bar chart — MAE & RMSE side by side
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results_df["Model"], y=results_df["MAE"],
        name="MAE", marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        x=results_df["Model"], y=results_df["RMSE"],
        name="RMSE", marker_color="#EF553B",
    ))
    fig.update_layout(
        barmode="group", height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Error (z-score units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_column_width=True)

    # Coverage 80% bar
    cov_df = results_df.dropna(subset=["Coverage80"])
    fig2 = go.Figure(go.Bar(
        x=cov_df["Model"], y=cov_df["Coverage80"],
        marker_color=["#00CC96" if v >= 80 else "#FFA15A" for v in cov_df["Coverage80"]],
        text=[f"{v:.1f}%" for v in cov_df["Coverage80"]], textposition="outside",
    ))
    fig2.add_hline(y=80, line_dash="dash", line_color="gray",
                   annotation_text="80% target", annotation_position="right")
    fig2.update_layout(
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Coverage 80% PI (%)", yaxis_range=[0, 100],
        title="Prediction Interval Coverage",
    )
    st.plotly_chart(fig2, use_column_width=True)

    st.dataframe(
        results_df.set_index("Model").style.highlight_min(
            subset=["MAE", "RMSE", "MAPE"], color="#d4edda"
        ).highlight_max(subset=["Coverage80"], color="#d4edda"),
        use_container_width=True,
    )


# ── Tab 4: Feature Importance ────────────────────────────────────────────────
with tab_feat:
    st.subheader("Feature Importance — Gradient-Boosted Models")

    for mname in ["LightGBM", "XGBoost"]:
        models = load_gbm_model(mname)
        if models is None:
            st.info(f"{mname} model not found — run `make train`.")
            continue

        if mname == "LightGBM":
            imp = models[0.5].feature_importances_
            names = models[0.5].booster_.feature_name()
        else:
            imp = models[0.5].feature_importances_
            names = models[0.5].feature_names_in_

        feat_df = pd.Series(imp, index=names).sort_values(ascending=False).head(20)

        fig = go.Figure(go.Bar(
            x=feat_df.values[::-1], y=feat_df.index[::-1],
            orientation="h", marker_color="#636EFA",
        ))
        fig.update_layout(
            title=f"{mname} — Top 20 Features",
            height=500, margin=dict(l=0, r=10, t=40, b=0),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, use_column_width=True)
