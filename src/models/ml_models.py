"""
Gradient-boosted tree models: XGBoost and LightGBM with quantile regression.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from loguru import logger
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

QUANTILES = [0.1, 0.5, 0.9]   # 80 % PI + median


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_target_split(
    df: pd.DataFrame,
    target_col: str = "load_MW",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostForecaster:
    def __init__(self):
        cfg = CFG["models"]["xgboost"]
        self.params = {
            "n_estimators": cfg["n_estimators"],
            "learning_rate": cfg["learning_rate"],
            "max_depth": cfg["max_depth"],
            "tree_method": "hist",
            "random_state": 42,
        }
        self.models: dict[float, xgb.XGBRegressor] = {}

    def fit(self, train: pd.DataFrame, target_col: str = "load_MW") -> XGBoostForecaster:
        X, y = _feature_target_split(train, target_col)
        for q in QUANTILES:
            logger.info(f"XGBoost: fitting quantile={q}")
            m = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                **self.params,
            )
            m.fit(X, y, verbose=False)
            self.models[q] = m
        logger.success("XGBoost fitted for all quantiles.")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {q: self.models[q].predict(X) for q in QUANTILES}
        return pd.DataFrame({
            "forecast": preds[0.5],
            "lower_80": preds[0.1],
            "upper_80": preds[0.9],
        }, index=X.index)

    def feature_importances(self) -> pd.Series:
        imp = self.models[0.5].feature_importances_
        return pd.Series(imp, index=self.models[0.5].feature_names_in_).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        joblib.dump(self.models, path)
        logger.info(f"XGBoost models saved → {path}")

    @classmethod
    def load(cls, path: Path) -> XGBoostForecaster:
        obj = cls.__new__(cls)
        obj.models = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMForecaster:
    def __init__(self):
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, train: pd.DataFrame, target_col: str = "load_MW") -> LightGBMForecaster:
        X, y = _feature_target_split(train, target_col)
        for q in QUANTILES:
            logger.info(f"LightGBM: fitting quantile={q}")
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                random_state=42,
                verbose=-1,
            )
            m.fit(X, y)
            self.models[q] = m
        logger.success("LightGBM fitted for all quantiles.")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {q: self.models[q].predict(X) for q in QUANTILES}
        return pd.DataFrame({
            "forecast": preds[0.5],
            "lower_80": preds[0.1],
            "upper_80": preds[0.9],
        }, index=X.index)

    def feature_importances(self) -> pd.Series:
        imp = self.models[0.5].feature_importances_
        names = self.models[0.5].booster_.feature_name()
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        joblib.dump(self.models, path)
        logger.info(f"LightGBM models saved → {path}")

    @classmethod
    def load(cls, path: Path) -> LightGBMForecaster:
        obj = cls.__new__(cls)
        obj.models = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.fetch_data import fetch_entsoe_load, fetch_openmeteo_weather
    from src.data.preprocess import build_features, train_test_split_temporal
    from src.evaluation.metrics import evaluate_forecasts

    load = fetch_entsoe_load()
    weather = fetch_openmeteo_weather()
    df = build_features(load, weather)
    train_df, test_df = train_test_split_temporal(df)

    for name, Cls, path in [
        ("xgboost", XGBoostForecaster, ROOT / "data" / "processed" / "xgboost.pkl"),
        ("lightgbm", LightGBMForecaster, ROOT / "data" / "processed" / "lightgbm.pkl"),
    ]:
        with mlflow.start_run(run_name=name):
            model = Cls()
            model.fit(train_df)
            X_test, y_test = _feature_target_split(test_df)
            preds = model.predict(X_test)
            metrics = evaluate_forecasts(y_test, preds["forecast"])
            mlflow.log_metrics(metrics)
            model.save(path)
