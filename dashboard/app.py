"""
Streamlit executive dashboard for AfriPower Forecast.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

st.set_page_config(
    page_title="AfriPower Forecast",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ AfriPower Forecast")
st.sidebar.markdown("Moroccan electricity demand intelligence")

model_choice = st.sidebar.selectbox(
    "Model",
    ["LightGBM", "XGBoost", "LSTM", "Prophet"],
    index=0,
)
horizon_days = st.sidebar.slider("Forecast horizon (days)", 1, 7, 7)
show_ci = st.sidebar.checkbox("Show confidence intervals", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data range**")
st.sidebar.markdown(f"Training: {CFG['data']['start_date']} → {CFG['data']['end_date']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_processed_data() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_model_results() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "model_results.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("⚡ AfriPower — African Energy Demand Intelligence")

tab_forecast, tab_eda, tab_models, tab_shap = st.tabs(
    ["📈 Forecast", "🔍 EDA", "🏆 Model Comparison", "🔬 Explainability"]
)

# --- Forecast tab ---
with tab_forecast:
    st.subheader(f"J+1 to J+{horizon_days} Forecast — {model_choice}")

    df = load_processed_data()
    if df is None:
        st.warning("No processed data found. Run `make data` then `make train` first.")
    else:
        from src.visualization.plots import forecast_ribbon_plotly

        # Mock forecast for display — replace with real model call after training
        last_known = df["load_MW"].iloc[-horizon_days * 24 :]
        future_idx = pd.date_range(
            last_known.index[-1] + pd.Timedelta("1h"),
            periods=horizon_days * 24,
            freq="1h",
            tz=last_known.index.tz,
        )
        mock_forecast = pd.DataFrame(
            {
                "forecast": last_known.mean() + (pd.Series(range(len(future_idx))) * 0),
                "lower_80": last_known.mean() * 0.93,
                "upper_80": last_known.mean() * 1.07,
                "lower_95": last_known.mean() * 0.88,
                "upper_95": last_known.mean() * 1.12,
            },
            index=future_idx,
        )

        fig = forecast_ribbon_plotly(
            df["load_MW"],
            mock_forecast,
            title=f"Morocco Load Forecast — {model_choice}",
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Peak forecast (MW)", f"{mock_forecast['upper_80'].max():,.0f}")
        col2.metric("Min forecast (MW)", f"{mock_forecast['lower_80'].min():,.0f}")
        col3.metric("Mean forecast (MW)", f"{mock_forecast['forecast'].mean():,.0f}")

# --- EDA tab ---
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    df = load_processed_data()
    if df is None:
        st.warning("No processed data found.")
    else:
        from src.visualization.plots import plot_hourly_profile
        import matplotlib.pyplot as plt

        st.markdown("**Load distribution by hour of day**")
        fig, ax = plt.subplots(figsize=(12, 4))
        profile = df["load_MW"].groupby(df.index.hour).mean()
        ax.bar(profile.index, profile.values, color="#1f77b4", alpha=0.8)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Mean Load (MW)")
        ax.set_xticks(range(24))
        st.pyplot(fig)

        st.markdown("**Raw statistics**")
        st.dataframe(df[["load_MW"] + [c for c in df.columns if "temp" in c]].describe())

# --- Model Comparison tab ---
with tab_models:
    st.subheader("Model Benchmark Results")
    results = load_model_results()
    if results is None:
        st.info("No results yet — run `make train` to populate this tab.")
    else:
        from src.visualization.plots import model_comparison_plotly

        fig = model_comparison_plotly(results)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results.set_index("model"))

# --- SHAP tab ---
with tab_shap:
    st.subheader("SHAP Feature Importance")
    shap_dir = ROOT / "data" / "processed" / "shap_plots"
    summary_img = shap_dir / "lightgbm_shap_summary.png"

    if summary_img.exists():
        st.image(str(summary_img), caption="LightGBM SHAP Summary", use_column_width=True)
    else:
        st.info("No SHAP plots found. Run `src/evaluation/explainability.py` after training.")
