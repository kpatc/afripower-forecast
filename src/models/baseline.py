"""
Baseline forecasting models: SARIMA and Prophet.
Trained on Tétouan city only (univariate, full-year 2017 dataset).
Target: load_norm (z-score normalised).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
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
        self.result = None

    def fit(self, train: pd.Series) -> SARIMAForecaster:
        logger.info(f"SARIMA — fitting on {len(train)} points …")
        model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = model.fit(disp=False)
        logger.success("SARIMA fitted.")
        return self

    def predict(self, steps: int) -> pd.DataFrame:
        fc = self.result.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)
        return pd.DataFrame({
            "forecast": mean,
            "lower_80": ci80.iloc[:, 0],
            "upper_80": ci80.iloc[:, 1],
            "lower_95": ci95.iloc[:, 0],
            "upper_95": ci95.iloc[:, 1],
        })

    def save(self, path: Path) -> None:
        # remove_data=True strips training arrays → shrinks file from ~3 GB to ~KB
        self.result.save(str(path), remove_data=True)
        logger.info(f"SARIMA saved → {path}")

    @classmethod
    def load(cls, path: Path) -> SARIMAForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        obj = cls.__new__(cls)
        obj.result = SARIMAXResults.load(str(path))
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
        df_p = train.reset_index()
        df_p.columns = ["ds", "y"]
        df_p["ds"] = df_p["ds"].dt.tz_localize(None)   # Prophet requires tz-naive
        logger.info(f"Prophet — fitting on {len(df_p)} points …")
        self.model.fit(df_p)
        logger.success("Prophet fitted.")
        return self

    def predict(self, steps: int, freq: str = "1h") -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        fc = self.model.predict(future).tail(steps).set_index("ds")
        return pd.DataFrame({
            "forecast": fc["yhat"],
            "lower_80": fc["yhat_lower"],
            "upper_80": fc["yhat_upper"],
        })

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Prophet saved → {path}")

    @classmethod
    def load(cls, path: Path) -> ProphetForecaster:
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj
