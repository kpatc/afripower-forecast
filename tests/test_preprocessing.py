"""
Unit tests for src/data/preprocess.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    add_calendar_features,
    add_city_dummies,
    add_holiday_feature,
    add_lag_features,
    add_rolling_features,
    clean_city_load,
    normalize_per_city,
    train_test_split_temporal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_series() -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=500, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    values = 3000 + 500 * np.sin(np.linspace(0, 4 * np.pi, 500)) + rng.normal(0, 50, 500)
    return pd.Series(values, index=idx, name="load_kW")


@pytest.fixture()
def multi_city_df() -> pd.DataFrame:
    """Minimal long-format DataFrame with two cities."""
    idx_a = pd.date_range("2023-01-01", periods=300, freq="1h", tz="UTC")
    idx_b = pd.date_range("2023-01-01", periods=300, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    df_a = pd.DataFrame({"load_kW": 2000 + rng.normal(0, 100, 300), "city": "alpha"}, index=idx_a)
    df_b = pd.DataFrame({"load_kW": 5000 + rng.normal(0, 200, 300), "city": "beta"},  index=idx_b)
    return pd.concat([df_a, df_b]).sort_index()


@pytest.fixture()
def calendar_df(multi_city_df) -> pd.DataFrame:
    return add_calendar_features(multi_city_df.copy())


# ---------------------------------------------------------------------------
# clean_city_load
# ---------------------------------------------------------------------------

def test_clean_removes_outliers(raw_series):
    dirty = raw_series.copy()
    dirty.iloc[10] = 1e9
    cleaned = clean_city_load(dirty)
    assert cleaned.iloc[10] < 1e9


def test_clean_interpolates_short_gaps(raw_series):
    gappy = raw_series.copy()
    gappy.iloc[20:24] = np.nan          # 4-hour gap → should be filled
    cleaned = clean_city_load(gappy)
    assert not cleaned.iloc[20:24].isna().any()


def test_clean_returns_series(raw_series):
    assert isinstance(clean_city_load(raw_series), pd.Series)


def test_clean_zeros_become_nan_then_filled(raw_series):
    with_zeros = raw_series.copy()
    with_zeros.iloc[5:8] = 0
    cleaned = clean_city_load(with_zeros)
    assert (cleaned.iloc[5:8] > 0).all()


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------

def test_calendar_columns_present(calendar_df):
    expected = ["hour", "day_of_week", "month", "is_weekend",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    for col in expected:
        assert col in calendar_df.columns, f"Missing: {col}"


def test_hour_range(calendar_df):
    assert calendar_df["hour"].between(0, 23).all()


def test_cyclical_bounds(calendar_df):
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert calendar_df[col].between(-1.01, 1.01).all(), f"{col} out of bounds"


def test_is_weekend_binary(calendar_df):
    assert set(calendar_df["is_weekend"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# add_holiday_feature
# ---------------------------------------------------------------------------

def test_holiday_column_added(calendar_df):
    df = add_holiday_feature(calendar_df.copy())
    assert "is_holiday" in df.columns


def test_holiday_binary(calendar_df):
    df = add_holiday_feature(calendar_df.copy())
    assert set(df["is_holiday"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# normalize_per_city
# ---------------------------------------------------------------------------

def test_normalize_zero_mean(multi_city_df):
    df_norm, stats = normalize_per_city(multi_city_df.copy())
    for city in multi_city_df["city"].unique():
        city_mean = df_norm.loc[df_norm["city"] == city, "load_norm"].mean()
        assert abs(city_mean) < 0.05, f"{city} mean not near 0: {city_mean}"


def test_normalize_returns_stats(multi_city_df):
    _, stats = normalize_per_city(multi_city_df.copy())
    assert "mean_kW" in stats.columns
    assert "std_kW" in stats.columns


def test_normalize_adds_load_norm(multi_city_df):
    df_norm, _ = normalize_per_city(multi_city_df.copy())
    assert "load_norm" in df_norm.columns


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------

def test_lag_columns_created(multi_city_df):
    df, _ = normalize_per_city(multi_city_df.copy())
    df = add_lag_features(df)
    assert "lag_1h" in df.columns
    assert "lag_24h" in df.columns


def test_lag_no_cross_city_leakage(multi_city_df):
    """lag_1h for city alpha at its first row must be NaN, not beta's last value."""
    df, _ = normalize_per_city(multi_city_df.copy())
    df = add_lag_features(df)
    first_alpha = df[df["city"] == "alpha"].iloc[0]
    assert pd.isna(first_alpha["lag_1h"])


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------

def test_rolling_columns_created(multi_city_df):
    df, _ = normalize_per_city(multi_city_df.copy())
    df = add_rolling_features(df)
    assert "roll_mean_24h" in df.columns
    assert "roll_std_24h" in df.columns


# ---------------------------------------------------------------------------
# add_city_dummies
# ---------------------------------------------------------------------------

def test_city_dummies_created(multi_city_df):
    df = add_city_dummies(multi_city_df.copy())
    assert "city_alpha" in df.columns
    assert "city_beta" in df.columns


def test_city_dummies_binary(multi_city_df):
    df = add_city_dummies(multi_city_df.copy())
    assert set(df["city_alpha"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# train_test_split_temporal
# ---------------------------------------------------------------------------

def test_split_ratio(multi_city_df):
    train, test = train_test_split_temporal(multi_city_df, test_ratio=0.2)
    assert abs(len(test) / len(multi_city_df) - 0.2) < 0.02


def test_split_no_overlap(multi_city_df):
    train, test = train_test_split_temporal(multi_city_df)
    assert train.index.max() <= test.index.min()


def test_split_full_coverage(multi_city_df):
    train, test = train_test_split_temporal(multi_city_df)
    assert len(train) + len(test) == len(multi_city_df)
