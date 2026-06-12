"""
Baseline forecasting models: SARIMA and Prophet.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
mlflow.set_experiment(CFG["mlflow"]["experiment_name"])


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

class SARIMAForecaster:
    def __init__(self):
        sarima_cfg = CFG["models"]["sarima"]
        self.order = tuple(sarima_cfg["order"])
        self.seasonal_order = tuple(sarima_cfg["seasonal_order"])
        self.model = None
        self.result = None

    def fit(self, train: pd.Series) -> SARIMAForecaster:
        logger.info("Fitting SARIMA model …")
        self.model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = self.model.fit(disp=False)
        logger.success("SARIMA fitted.")
        return self

    def predict(self, steps: int) -> pd.DataFrame:
        forecast = self.result.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.2)   # 80 % CI
        ci95 = forecast.conf_int(alpha=0.05) # 95 % CI
        return pd.DataFrame({
            "forecast": mean,
            "lower_80": ci.iloc[:, 0],
            "upper_80": ci.iloc[:, 1],
            "lower_95": ci95.iloc[:, 0],
            "upper_95": ci95.iloc[:, 1],
        })

    def save(self, path: Path) -> None:
        joblib.dump(self.result, path)
        logger.info(f"SARIMA model saved → {path}")

    @classmethod
    def load(cls, path: Path) -> SARIMAForecaster:
        obj = cls()
        obj.result = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

class ProphetForecaster:
    def __init__(self):
        self.model = Prophet(
            interval_width=0.80,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    def fit(self, train: pd.Series) -> ProphetForecaster:
        df_prophet = train.reset_index()
        df_prophet.columns = ["ds", "y"]
        logger.info("Fitting Prophet model …")
        self.model.fit(df_prophet)
        logger.success("Prophet fitted.")
        return self

    def predict(self, steps: int, freq: str = "1h") -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future).tail(steps).set_index("ds")
        return pd.DataFrame({
            "forecast": forecast["yhat"],
            "lower_80": forecast["yhat_lower"],
            "upper_80": forecast["yhat_upper"],
        })

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Prophet model saved → {path}")

    @classmethod
    def load(cls, path: Path) -> ProphetForecaster:
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.preprocess import build_features, train_test_split_temporal
    from src.data.fetch_data import fetch_entsoe_load, fetch_openmeteo_weather
    from src.evaluation.metrics import evaluate_forecasts

    load = fetch_entsoe_load()
    weather = fetch_openmeteo_weather()
    df = build_features(load, weather)
    train_df, test_df = train_test_split_temporal(df)

    horizon = CFG["data"]["forecast_horizon"] * 24

    with mlflow.start_run(run_name="sarima"):
        sarima = SARIMAForecaster()
        sarima.fit(train_df["load_MW"])
        preds = sarima.predict(horizon)
        metrics = evaluate_forecasts(test_df["load_MW"].iloc[:horizon], preds["forecast"])
        mlflow.log_params({"order": sarima.order, "seasonal_order": sarima.seasonal_order})
        mlflow.log_metrics(metrics)
        sarima.save(ROOT / "data" / "processed" / "sarima.pkl")

    with mlflow.start_run(run_name="prophet"):
        prophet = ProphetForecaster()
        prophet.fit(train_df["load_MW"])
        preds = prophet.predict(horizon)
        metrics = evaluate_forecasts(test_df["load_MW"].iloc[:horizon], preds["forecast"])
        mlflow.log_metrics(metrics)
        prophet.save(ROOT / "data" / "processed" / "prophet.pkl")
