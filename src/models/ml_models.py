"""
Gradient-boosted tree models: XGBoost and LightGBM with quantile regression.
Target: load_norm (z-score normalised per city).
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

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

TARGET = "load_norm"
QUANTILES = [0.1, 0.5, 0.9]   # 80% PI (q10/q90) + median


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostForecaster:
    def __init__(self):
        cfg = CFG["models"]["xgboost"]
        self.base_params = {
            "n_estimators":  cfg["n_estimators"],
            "learning_rate": cfg["learning_rate"],
            "max_depth":     cfg["max_depth"],
            "tree_method":   "hist",
            "random_state":  42,
        }
        self.models: dict[float, xgb.XGBRegressor] = {}

    def fit(self, train: pd.DataFrame) -> XGBoostForecaster:
        X, y = _split(train)
        for q in QUANTILES:
            logger.info(f"XGBoost — fitting quantile q={q}")
            m = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                **self.base_params,
            )
            m.fit(X, y, verbose=False)
            self.models[q] = m
        logger.success("XGBoost fitted (q=0.1, 0.5, 0.9).")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "forecast":  self.models[0.5].predict(X),
            "lower_80":  self.models[0.1].predict(X),
            "upper_80":  self.models[0.9].predict(X),
        }, index=X.index)

    def feature_importances(self) -> pd.Series:
        imp = self.models[0.5].feature_importances_
        return pd.Series(imp, index=self.models[0.5].feature_names_in_).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        joblib.dump(self.models, path)
        logger.info(f"XGBoost saved → {path}")

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

    def fit(self, train: pd.DataFrame) -> LightGBMForecaster:
        X, y = _split(train)
        for q in QUANTILES:
            logger.info(f"LightGBM — fitting quantile q={q}")
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
        logger.success("LightGBM fitted (q=0.1, 0.5, 0.9).")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "forecast":  self.models[0.5].predict(X),
            "lower_80":  self.models[0.1].predict(X),
            "upper_80":  self.models[0.9].predict(X),
        }, index=X.index)

    def feature_importances(self) -> pd.Series:
        imp = self.models[0.5].feature_importances_
        names = self.models[0.5].booster_.feature_name()
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        joblib.dump(self.models, path)
        logger.info(f"LightGBM saved → {path}")

    @classmethod
    def load(cls, path: Path) -> LightGBMForecaster:
        obj = cls.__new__(cls)
        obj.models = joblib.load(path)
        return obj
