"""
Unit tests for src/data/preprocess.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    add_calendar_features,
    add_holiday_feature,
    add_lag_features,
    add_rolling_features,
    clean_load,
    train_test_split_temporal,
)


@pytest.fixture()
def sample_load() -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=500, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    values = 3000 + 500 * np.sin(np.linspace(0, 4 * np.pi, 500)) + rng.normal(0, 50, 500)
    return pd.Series(values, index=idx, name="load_MW")


@pytest.fixture()
def sample_df(sample_load) -> pd.DataFrame:
    df = sample_load.to_frame("load_MW")
    df = add_calendar_features(df)
    return df


# ---------------------------------------------------------------------------
# clean_load
# ---------------------------------------------------------------------------

def test_clean_load_removes_outliers(sample_load):
    dirty = sample_load.copy()
    dirty.iloc[10] = 1e7  # inject outlier
    cleaned = clean_load(dirty)
    assert cleaned.iloc[10] < 1e7


def test_clean_load_interpolates_gaps(sample_load):
    gappy = sample_load.copy()
    gappy.iloc[20:24] = np.nan
    cleaned = clean_load(gappy)
    assert not cleaned.isna().any()


def test_clean_load_returns_series(sample_load):
    result = clean_load(sample_load)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------

def test_calendar_features_present(sample_df):
    expected = ["hour", "day_of_week", "month", "is_weekend",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    for col in expected:
        assert col in sample_df.columns, f"Missing column: {col}"


def test_hour_range(sample_df):
    assert sample_df["hour"].between(0, 23).all()


def test_cyclical_encoding_bounds(sample_df):
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert sample_df[col].between(-1, 1).all(), f"{col} out of [-1, 1]"


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------

def test_lag_features_created(sample_df):
    df = add_lag_features(sample_df.copy())
    assert "lag_1h" in df.columns
    assert "lag_168h" in df.columns


def test_lag_values_correct(sample_df):
    df = add_lag_features(sample_df.copy())
    # lag_1h at row i should equal load_MW at row i-1
    assert df["lag_1h"].iloc[5] == pytest.approx(df["load_MW"].iloc[4])


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------

def test_rolling_features_created(sample_df):
    df = add_rolling_features(sample_df.copy())
    assert "roll_mean_24h" in df.columns
    assert "roll_std_24h" in df.columns


# ---------------------------------------------------------------------------
# train_test_split_temporal
# ---------------------------------------------------------------------------

def test_split_ratio(sample_df):
    train, test = train_test_split_temporal(sample_df, test_ratio=0.2)
    assert abs(len(test) / len(sample_df) - 0.2) < 0.01


def test_split_no_overlap(sample_df):
    train, test = train_test_split_temporal(sample_df)
    assert train.index.max() < test.index.min()


def test_split_covers_full_dataset(sample_df):
    train, test = train_test_split_temporal(sample_df)
    assert len(train) + len(test) == len(sample_df)
