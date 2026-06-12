"""
SHAP-based model explainability for tree and deep learning models.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "data" / "processed" / "shap_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_shap_tree(
    model,
    X: pd.DataFrame,
    model_name: str = "model",
    max_samples: int = 500,
) -> shap.Explanation:
    X_sample = X.sample(min(max_samples, len(X)), random_state=42)
    logger.info(f"Computing SHAP values for {model_name} ({len(X_sample)} samples) …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values


def plot_summary(
    shap_values: shap.Explanation,
    model_name: str = "model",
    max_display: int = 20,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, show=False, max_display=max_display)
    path = PLOTS_DIR / f"{model_name}_shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"SHAP summary saved → {path}")


def plot_waterfall(
    shap_values: shap.Explanation,
    sample_idx: int = 0,
    model_name: str = "model",
) -> None:
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    path = PLOTS_DIR / f"{model_name}_shap_waterfall_{sample_idx}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"SHAP waterfall saved → {path}")


def plot_dependence(
    shap_values: shap.Explanation,
    feature: str,
    model_name: str = "model",
) -> None:
    shap.dependence_plot(feature, shap_values.values, shap_values.data, show=False)
    path = PLOTS_DIR / f"{model_name}_shap_dep_{feature}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"SHAP dependence plot saved → {path}")


def top_features(shap_values: shap.Explanation, n: int = 10) -> pd.Series:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.Series(mean_abs, index=shap_values.feature_names)
        .sort_values(ascending=False)
        .head(n)
    )
