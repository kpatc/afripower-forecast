"""
Cleaning, feature engineering, and normalization for the AfriPower dataset.

Input  : combined long-format DataFrame from fetch_data.load_all_sources()
Output : feature matrix ready for ML training (data/processed/features.parquet)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

LAGS: list[int] = CFG["features"]["lags"]
WINDOWS: list[int] = CFG["features"]["rolling_windows"]

# Cities with known Ampère→kW conversion (less reliable absolute scale)
AMPERE_CITIES = {"laayoune", "boujdour", "foum_eloued"}


# ---------------------------------------------------------------------------
# Cleaning per city
# ---------------------------------------------------------------------------

def clean_city_load(series: pd.Series) -> pd.Series:
    """
    For a single city's hourly load series:
    - Remove zeros (meter outages)
    - Clip outliers beyond 4σ
    - Interpolate gaps ≤ 6h
    """
    s = series.copy()
    s = s.replace(0, np.nan)
    mu, sigma = s.mean(), s.std()
    s = s.clip(lower=mu - 4 * sigma, upper=mu + 4 * sigma)
    s = s.interpolate(method="time", limit=6)
    return s


def clean_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_city_load per city, return cleaned DataFrame."""
    cleaned = []
    for city, group in df.groupby("city", sort=False):
        group = group.copy()
        group["load_kW"] = clean_city_load(group["load_kW"])
        cleaned.append(group)
    result = pd.concat(cleaned).sort_index()
    result = result.dropna(subset=["load_kW"])
    return result


# ---------------------------------------------------------------------------
# Per-city z-score normalisation
# ---------------------------------------------------------------------------

def normalize_per_city(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Z-score normalize load_kW per city so all cities are on the same scale.
    Returns (normalized_df, stats_df) where stats_df has columns [mean, std] per city.
    """
    stats_rows = []
    frames = []

    for city, group in df.groupby("city", sort=False):
        mu = group["load_kW"].mean()
        sigma = group["load_kW"].std()
        group = group.copy()
        group["load_norm"] = (group["load_kW"] - mu) / (sigma + 1e-8)
        stats_rows.append({"city": city, "mean_kW": mu, "std_kW": sigma})
        frames.append(group)

    stats = pd.DataFrame(stats_rows).set_index("city")
    result = pd.concat(frames).sort_index()
    return result, stats


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df = df.copy()
    df["hour"]        = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"]       = idx.month
    df["day_of_year"] = idx.dayofyear
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    # Cyclical encodings
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ---------------------------------------------------------------------------
# Holidays (Moroccan public holidays)
# ---------------------------------------------------------------------------

def load_holidays(path: Path | None = None) -> pd.DatetimeIndex:
    path = path or ROOT / "data" / "external" / "ma_holidays.csv"
    if not path.exists():
        logger.warning("Holiday file not found — is_holiday set to 0 everywhere.")
        return pd.DatetimeIndex([])
    hdf = pd.read_csv(path, parse_dates=["date"])
    return pd.DatetimeIndex(hdf["date"])


def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    holidays = load_holidays()
    # Normalize index to date (tz-naive) for comparison with tz-naive holiday dates
    normalized = df.index.normalize()
    if normalized.tz is not None:
        normalized = normalized.tz_localize(None)
    df = df.copy()
    df["is_holiday"] = normalized.isin(holidays).astype(int)
    return df


# ---------------------------------------------------------------------------
# Lag & rolling features  (computed per city to avoid cross-city leakage)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, target_col: str = "load_norm") -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"lag_{lag}h"] = df.groupby("city")[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "load_norm") -> pd.DataFrame:
    df = df.copy()
    for window in WINDOWS:
        shifted = df.groupby("city")[target_col].shift(1)
        df[f"roll_mean_{window}h"] = shifted.groupby(df["city"]).transform(
            lambda x: x.rolling(window, min_periods=window // 2).mean()
        )
        df[f"roll_std_{window}h"] = shifted.groupby(df["city"]).transform(
            lambda x: x.rolling(window, min_periods=window // 2).std()
        )
    return df


# ---------------------------------------------------------------------------
# City as categorical feature (one-hot)
# ---------------------------------------------------------------------------

def add_city_dummies(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)
    return pd.concat([df, dummies], axis=1)


# ---------------------------------------------------------------------------
# Attach Open-Meteo weather (for UCI cities)
# ---------------------------------------------------------------------------

def attach_weather(
    df: pd.DataFrame,
    weather_by_city: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Attach weather data per city.
    Tetouan already has weather embedded; UCI cities get Open-Meteo data.
    Drops NaN weather columns inherited from concat before joining.
    """
    frames = []
    for city, group in df.groupby("city", sort=False):
        group = group.copy()
        if city in weather_by_city:
            w = weather_by_city[city].resample("1h").mean()
            # Drop columns that are already present (NaN placeholders from concat)
            overlap = [c for c in w.columns if c in group.columns]
            group = group.drop(columns=overlap)
            group = group.join(w, how="left")
        frames.append(group)
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(
    combined: pd.DataFrame,
    weather_by_city: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    combined : long-format DataFrame from load_all_sources()
    weather_by_city : optional dict city → Open-Meteo weather DataFrame

    Returns
    -------
    (features_df, normalization_stats)
    features_df is saved to data/processed/features.parquet
    """
    logger.info("Starting feature engineering pipeline …")

    df = clean_all(combined)
    df, norm_stats = normalize_per_city(df)

    df = add_calendar_features(df)
    df = add_holiday_feature(df)

    if weather_by_city:
        df = attach_weather(df, weather_by_city)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_city_dummies(df)

    # Drop raw columns not used as features
    drop_cols = ["city", "load_kW"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Only require lag/rolling + target to be non-null (weather is optional)
    required_cols = (
        ["load_norm"]
        + [c for c in df.columns if c.startswith("lag_")]
        + [c for c in df.columns if c.startswith("roll_")]
    )
    df = df.dropna(subset=required_cols)

    # Fill remaining NaN (weather columns absent for some cities) with 0
    df = df.fillna(0)

    logger.info(f"Feature matrix: {df.shape}  "
                f"({df.shape[0]} rows × {df.shape[1]} cols)")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "features.parquet")
    norm_stats.to_csv(out_dir / "norm_stats.csv")
    logger.success("Saved → data/processed/features.parquet + norm_stats.csv")

    return df, norm_stats


# ---------------------------------------------------------------------------
# Train / test split  (temporal, respects city ordering)
# ---------------------------------------------------------------------------

def train_test_split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ---------------------------------------------------------------------------
# Inverse transform predictions back to kW
# ---------------------------------------------------------------------------

def denormalize(
    values: np.ndarray,
    city: str,
    norm_stats: pd.DataFrame,
) -> np.ndarray:
    mu = norm_stats.loc[city, "mean_kW"]
    sigma = norm_stats.loc[city, "std_kW"]
    return values * sigma + mu
