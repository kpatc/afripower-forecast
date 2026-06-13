"""
Full training pipeline — trains all 5 models and logs results to MLflow.

Usage:
    python -m src.train_all
    or: make train
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
mlflow.set_experiment(CFG["mlflow"]["experiment_name"])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    from src.data.fetch_data import load_all_sources, fetch_openmeteo_weather
    from src.data.preprocess import build_features, train_test_split_temporal

    logger.info("Loading raw data …")
    combined = load_all_sources()

    CITY_RANGES = {
        "laayoune":    ("2022-09-14", "2024-05-24"),
        "boujdour":    ("2022-09-14", "2024-05-24"),
        "foum_eloued": ("2022-09-14", "2024-05-24"),
        "marrakech":   ("2023-01-09", "2024-01-09"),
    }
    weather_by_city = {
        city: fetch_openmeteo_weather(city=city, start=s, end=e)
        for city, (s, e) in CITY_RANGES.items()
    }

    features, norm_stats = build_features(combined, weather_by_city=weather_by_city)
    train, test = train_test_split_temporal(features, test_ratio=0.15)

    logger.info(f"Train: {len(train):,}  Test: {len(test):,}  Features: {features.shape[1]}")
    return train, test, norm_stats, combined


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def log_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from src.evaluation.metrics import evaluate_forecasts
    metrics = evaluate_forecasts(y_true, y_pred)
    mlflow.log_metrics(metrics)
    return metrics


# ---------------------------------------------------------------------------
# 1 & 2 — SARIMA + Prophet  (Tétouan only, univariate)
# ---------------------------------------------------------------------------

def train_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    from src.models.baseline import SARIMAForecaster, ProphetForecaster
    from src.evaluation.metrics import evaluate_forecasts

    # Tetouan (2017) sits entirely in the global training split because UCI cities
    # (2022-2024) occupy the chronologically-later test rows.  Do a separate 85/15
    # temporal split on Tetouan's own rows so the baselines have a proper test set.
    all_tet = pd.concat([train_df, test_df])
    tet_series = all_tet.loc[all_tet["city_tetouan"] == 1, "load_norm"].sort_index()
    split_idx  = int(len(tet_series) * 0.85)
    train_tet  = tet_series.iloc[:split_idx]
    test_tet   = tet_series.iloc[split_idx:]
    horizon    = min(CFG["data"]["forecast_horizon"] * 24, len(test_tet))

    # --- SARIMA ---
    with mlflow.start_run(run_name="sarima"):
        mlflow.log_params({"order": CFG["models"]["sarima"]["order"],
                           "seasonal_order": CFG["models"]["sarima"]["seasonal_order"],
                           "city": "tetouan"})
        sarima = SARIMAForecaster()
        sarima.fit(train_tet)
        preds  = sarima.predict(horizon)
        metrics = evaluate_forecasts(test_tet.values[:horizon], preds["forecast"].values)
        mlflow.log_metrics(metrics)
        sarima.save(MODELS_DIR / "sarima.pkl")
        logger.success(f"SARIMA → {metrics}")

    # --- Prophet ---
    with mlflow.start_run(run_name="prophet"):
        mlflow.log_params({"city": "tetouan"})
        prophet = ProphetForecaster()
        prophet.fit(train_tet)
        preds   = prophet.predict(horizon)
        metrics = evaluate_forecasts(test_tet.values[:horizon], preds["forecast"].values)
        mlflow.log_metrics(metrics)
        prophet.save(MODELS_DIR / "prophet.pkl")
        logger.success(f"Prophet → {metrics}")


# ---------------------------------------------------------------------------
# 3 & 4 — XGBoost + LightGBM  (full multi-city)
# ---------------------------------------------------------------------------

def train_ml_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    from src.models.ml_models import XGBoostForecaster, LightGBMForecaster, _split
    from src.evaluation.metrics import evaluate_forecasts

    X_test, y_test = _split(test_df)

    for name, Cls, path in [
        ("xgboost",  XGBoostForecaster,  MODELS_DIR / "xgboost.pkl"),
        ("lightgbm", LightGBMForecaster, MODELS_DIR / "lightgbm.pkl"),
    ]:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"cities": "all_5", "target": "load_norm"})
            model = Cls()
            model.fit(train_df)
            preds   = model.predict(X_test)
            metrics = evaluate_forecasts(
                y_test.values, preds["forecast"].values,
                lower_80=preds["lower_80"].values,
                upper_80=preds["upper_80"].values,
            )
            mlflow.log_metrics(metrics)
            model.save(path)
            logger.success(f"{name.upper()} → {metrics}")


# ---------------------------------------------------------------------------
# 5 — LSTM  (full multi-city)
# ---------------------------------------------------------------------------

def train_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    from src.models.deep_learning import LSTMForecaster
    from src.evaluation.metrics import evaluate_forecasts

    horizon = CFG["data"]["forecast_horizon"] * 24

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params({**CFG["models"]["lstm"], "cities": "all_5"})
        model = LSTMForecaster(input_size=train_df.shape[1], horizon=horizon)
        model.fit(train_df, epochs=30)

        preds_raw = model.predict(test_df)
        y_true    = test_df["load_norm"].values[:horizon]
        metrics   = evaluate_forecasts(y_true, preds_raw[:len(y_true)])
        mlflow.log_metrics(metrics)
        model.save(MODELS_DIR / "lstm.pt")
        logger.success(f"LSTM → {metrics}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df, test_df, norm_stats, combined = load_data()

    logger.info("=" * 60)
    logger.info("STEP 1/3 — Baseline models (SARIMA + Prophet) on Tétouan")
    logger.info("=" * 60)
    train_baselines(train_df, test_df)

    logger.info("=" * 60)
    logger.info("STEP 2/3 — ML models (XGBoost + LightGBM) on all cities")
    logger.info("=" * 60)
    train_ml_models(train_df, test_df)

    logger.info("=" * 60)
    logger.info("STEP 3/3 — LSTM on all cities")
    logger.info("=" * 60)
    train_lstm(train_df, test_df)

    logger.success("All models trained. Run: mlflow ui --backend-store-uri mlflow_runs")
