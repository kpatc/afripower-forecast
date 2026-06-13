"""
SHAP explainability — LightGBM + XGBoost on the test split.
Outputs PNG files to assets/shap_*.png

Usage:
    python -m src.evaluation.run_shap
    or: make shap
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
ASSETS    = ROOT / "assets"

TARGET = "load_norm"
TEST_RATIO = 0.15
N_SHAP_SAMPLES = 500   # TreeExplainer is exact; cap for speed


def load_test_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    return df.iloc[int(n * (1 - TEST_RATIO)):]


def shap_for_model(name: str, df: pd.DataFrame) -> None:
    path = PROCESSED / f"{name}.pkl"
    if not path.exists():
        logger.warning(f"{name}.pkl not found — skipping.")
        return

    models = joblib.load(path)
    median_model = models[0.5]

    X_test = load_test_split(df).drop(columns=[TARGET], errors="ignore")
    X_sample = X_test.sample(min(N_SHAP_SAMPLES, len(X_test)), random_state=42)

    logger.info(f"Computing SHAP for {name.upper()} ({len(X_sample)} samples) …")
    explainer   = shap.TreeExplainer(median_model)
    shap_values = explainer(X_sample)

    # ── 1. Summary (beeswarm) ────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f"{name.upper()} — SHAP Feature Impact (top 20)", fontsize=13, pad=12)
    plt.tight_layout()
    out = ASSETS / f"shap_{name}_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved → {out}")

    # ── 2. Bar (mean |SHAP|) ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f"{name.upper()} — Mean |SHAP| Value (top 20)", fontsize=13, pad=12)
    plt.tight_layout()
    out = ASSETS / f"shap_{name}_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved → {out}")

    # ── 3. Waterfall (peak-load sample) ─────────────────────────────────────
    peak_idx = int(X_sample["load_norm_lag_1"].abs().argmax()) if "load_norm_lag_1" in X_sample.columns else 0
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[peak_idx], show=False, max_display=15)
    plt.title(f"{name.upper()} — SHAP Waterfall (peak-load sample)", fontsize=13, pad=12)
    plt.tight_layout()
    out = ASSETS / f"shap_{name}_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved → {out}")

    # Print top-10 features
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top = pd.Series(mean_abs, index=shap_values.feature_names).sort_values(ascending=False).head(10)
    logger.info(f"\n{name.upper()} top-10 features:\n{top.to_string()}")


if __name__ == "__main__":
    feat_path = PROCESSED / "features.parquet"
    if not feat_path.exists():
        logger.error("features.parquet not found — run `make train` first.")
        sys.exit(1)

    df = pd.read_parquet(feat_path)
    logger.info(f"Features loaded: {df.shape}")

    for model_name in ("lightgbm", "xgboost"):
        shap_for_model(model_name, df)

    logger.success("SHAP analysis complete. Plots saved to assets/")
