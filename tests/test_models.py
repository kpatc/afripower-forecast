"""
Unit tests for ML models — XGBoost and LightGBM interface contracts.
Tests use synthetic data; no trained model files required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ml_models import LightGBMForecaster, XGBoostForecaster, _split

TARGET = "load_norm"
N_ROWS = 300
N_FEATURES = 10


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=N_ROWS, freq="1h", tz="UTC")
    data = {f"feat_{i}": rng.normal(0, 1, N_ROWS) for i in range(N_FEATURES)}
    data[TARGET] = rng.normal(0, 1, N_ROWS)
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def xgb_fitted(synthetic_df) -> XGBoostForecaster:
    m = XGBoostForecaster()
    m.fit(synthetic_df)
    return m


@pytest.fixture()
def lgbm_fitted(synthetic_df) -> LightGBMForecaster:
    m = LightGBMForecaster()
    m.fit(synthetic_df)
    return m


# ---------------------------------------------------------------------------
# _split helper
# ---------------------------------------------------------------------------

def test_split_drops_target(synthetic_df):
    X, y = _split(synthetic_df)
    assert TARGET not in X.columns
    assert y.name == TARGET


def test_split_shapes(synthetic_df):
    X, y = _split(synthetic_df)
    assert len(X) == len(y) == N_ROWS
    assert X.shape[1] == N_FEATURES


# ---------------------------------------------------------------------------
# XGBoostForecaster
# ---------------------------------------------------------------------------

def test_xgb_fit_returns_self(synthetic_df):
    m = XGBoostForecaster()
    result = m.fit(synthetic_df)
    assert result is m


def test_xgb_predict_shape(xgb_fitted, synthetic_df):
    X, _ = _split(synthetic_df)
    preds = xgb_fitted.predict(X)
    assert len(preds) == N_ROWS


def test_xgb_predict_columns(xgb_fitted, synthetic_df):
    X, _ = _split(synthetic_df)
    preds = xgb_fitted.predict(X)
    assert set(preds.columns) == {"forecast", "lower_80", "upper_80"}


def test_xgb_intervals_ordered(xgb_fitted, synthetic_df):
    # Independent quantile models can cross on pure noise; check lower ≤ upper on average
    X, _ = _split(synthetic_df)
    preds = xgb_fitted.predict(X)
    assert (preds["lower_80"] <= preds["upper_80"]).mean() > 0.9


def test_xgb_feature_importances(xgb_fitted):
    imp = xgb_fitted.feature_importances()
    assert len(imp) == N_FEATURES
    assert (imp >= 0).all()


def test_xgb_save_load(xgb_fitted, synthetic_df, tmp_path):
    path = tmp_path / "xgb.pkl"
    xgb_fitted.save(path)
    loaded = XGBoostForecaster.load(path)
    X, _ = _split(synthetic_df)
    preds_orig   = xgb_fitted.predict(X)["forecast"].values
    preds_loaded = loaded.predict(X)["forecast"].values
    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


# ---------------------------------------------------------------------------
# LightGBMForecaster
# ---------------------------------------------------------------------------

def test_lgbm_fit_returns_self(synthetic_df):
    m = LightGBMForecaster()
    result = m.fit(synthetic_df)
    assert result is m


def test_lgbm_predict_shape(lgbm_fitted, synthetic_df):
    X, _ = _split(synthetic_df)
    preds = lgbm_fitted.predict(X)
    assert len(preds) == N_ROWS


def test_lgbm_predict_columns(lgbm_fitted, synthetic_df):
    X, _ = _split(synthetic_df)
    preds = lgbm_fitted.predict(X)
    assert set(preds.columns) == {"forecast", "lower_80", "upper_80"}


def test_lgbm_intervals_ordered(lgbm_fitted, synthetic_df):
    X, _ = _split(synthetic_df)
    preds = lgbm_fitted.predict(X)
    assert (preds["lower_80"] <= preds["upper_80"]).all()


def test_lgbm_feature_importances(lgbm_fitted):
    imp = lgbm_fitted.feature_importances()
    assert len(imp) == N_FEATURES
    assert (imp >= 0).all()


def test_lgbm_save_load(lgbm_fitted, synthetic_df, tmp_path):
    path = tmp_path / "lgbm.pkl"
    lgbm_fitted.save(path)
    loaded = LightGBMForecaster.load(path)
    X, _ = _split(synthetic_df)
    preds_orig   = lgbm_fitted.predict(X)["forecast"].values
    preds_loaded = loaded.predict(X)["forecast"].values
    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)
