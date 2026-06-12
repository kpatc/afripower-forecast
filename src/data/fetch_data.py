"""
Data loading from local files + Open-Meteo weather API.

Sources:
  - data/raw/powerconsumption.csv   → Tetouan 2017 (10-min, kW, 3 zones + weather)
  - data/raw/Data Morocco.xlsx      → UCI Smart Meters 2022-2024
                                      Laayoune/Boujdour/Foum Eloued (10-min, Ampères)
                                      Marrakech (30-min, kW)
"""
from __future__ import annotations

from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
import yaml
from loguru import logger
from retry_requests import retry

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# City coordinates for Open-Meteo
# ---------------------------------------------------------------------------

CITY_COORDS: dict[str, tuple[float, float]] = {
    "tetouan":    (35.5785, -5.3684),
    "laayoune":   (27.1536, -13.2033),
    "boujdour":   (26.1333, -14.4833),
    "foum_eloued":(27.9802, -12.8253),
    "marrakech":  (31.6295, -7.9811),
}

# V=220V, PF=0.9 → P(kW) = 220 × I(A) × 0.9 / 1000
AMPERE_TO_KW = 220 * 0.9 / 1000


# ---------------------------------------------------------------------------
# Source 1 : Tetouan (CSV)
# ---------------------------------------------------------------------------

def load_tetouan() -> dict[str, pd.Series | pd.DataFrame]:
    """
    Returns:
        {
          "load": pd.Series  (hourly, kW, index=DatetimeIndex UTC),
          "weather": pd.DataFrame (hourly weather columns already in the file)
        }
    """
    path = RAW_DIR / "powerconsumption.csv"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").set_index("Datetime")
    df.index = df.index.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")

    # Resample 10-min → hourly mean
    df = df.resample("1h").mean()

    # Total load = sum of 3 zones (already in kW)
    load = (
        df[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]]
        .sum(axis=1)
        .rename("load_kW")
    )

    # Weather already in file
    weather_cols = ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows"]
    weather = df[weather_cols].copy()
    weather.columns = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                       "shortwave_radiation", "diffuse_radiation"]

    logger.success(f"Tetouan loaded: {len(load)} hourly points "
                   f"[{load.index[0].date()} → {load.index[-1].date()}]")
    return {"load": load, "weather": weather}


# ---------------------------------------------------------------------------
# Source 2 : UCI Smart Meters (Excel)
# ---------------------------------------------------------------------------

_SHEET_META = {
    "Laayoune":    {"unit": "A", "city": "laayoune"},
    "Boujdour":    {"unit": "A", "city": "boujdour"},
    "Foum eloued": {"unit": "A", "city": "foum_eloued"},
    "Marrakech":   {"unit": "kW", "city": "marrakech"},
}


def load_uci_smartmeters() -> dict[str, pd.Series]:
    """
    Returns a dict  city_name → pd.Series (hourly, kW, index=DatetimeIndex UTC).
    Ampère zones are converted: P(kW) = 220V × I(A) × 0.9 / 1000.
    All zones are summed to give city-level load.
    """
    path = RAW_DIR / "Data Morocco.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading UCI Smart Meters Excel (this takes ~30s) …")
    xl = pd.ExcelFile(path, engine="openpyxl")
    result: dict[str, pd.Series] = {}

    for sheet, meta in _SHEET_META.items():
        df = xl.parse(sheet, parse_dates=["DateTime"])
        df = df.sort_values("DateTime").set_index("DateTime")
        df.index = df.index.tz_localize("UTC", ambiguous="infer",
                                         nonexistent="shift_forward")

        zone_cols = [c for c in df.columns if c.startswith("zone")]

        if meta["unit"] == "A":
            df[zone_cols] = df[zone_cols] * AMPERE_TO_KW

        total = df[zone_cols].sum(axis=1)

        # Resample to hourly (10-min or 30-min → 1h mean)
        total = total.resample("1h").mean()
        total.name = "load_kW"

        city = meta["city"]
        result[city] = total
        logger.success(f"{sheet} loaded: {len(total)} hourly points "
                       f"[{total.index[0].date()} → {total.index[-1].date()}]")

    return result


# ---------------------------------------------------------------------------
# Combine all sources into one DataFrame
# ---------------------------------------------------------------------------

def load_all_sources() -> pd.DataFrame:
    """
    Returns a single long-format DataFrame with columns:
      load_kW, city  (index = DatetimeIndex UTC, hourly)
    Ready for feature engineering.
    """
    frames: list[pd.DataFrame] = []

    # Tetouan
    try:
        t = load_tetouan()
        df_t = t["load"].to_frame()
        df_t["city"] = "tetouan"
        # attach embedded weather
        df_t = df_t.join(t["weather"], how="left")
        frames.append(df_t)
    except FileNotFoundError as e:
        logger.warning(str(e))

    # UCI multi-city
    try:
        uci = load_uci_smartmeters()
        for city, series in uci.items():
            df_c = series.to_frame()
            df_c["city"] = city
            frames.append(df_c)
    except FileNotFoundError as e:
        logger.warning(str(e))

    if not frames:
        raise RuntimeError("No data source could be loaded. Check data/raw/.")

    combined = pd.concat(frames, axis=0).sort_index()
    logger.success(f"All sources combined: {len(combined)} hourly rows, "
                   f"cities={combined['city'].unique().tolist()}")
    return combined


# ---------------------------------------------------------------------------
# Open-Meteo — historical weather (for UCI cities, 2022-2024)
# ---------------------------------------------------------------------------

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
]


def fetch_openmeteo_weather(
    city: str = "marrakech",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly historical weather for a given city from Open-Meteo archive."""
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Choose from {list(CITY_COORDS)}")

    lat, lon = CITY_COORDS[city]
    start = start or "2022-09-14"
    end = end or "2024-05-24"

    out_path = RAW_DIR / f"weather_{city}_{start}_{end}.parquet"
    if out_path.exists():
        logger.info(f"Weather cache hit: {out_path.name}")
        return pd.read_parquet(out_path)

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om_client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": WEATHER_VARIABLES,
        "timezone": "UTC",
    }

    logger.info(f"Fetching Open-Meteo weather for {city} [{start} → {end}]")
    responses = om_client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    response = responses[0]
    hourly = response.Hourly()

    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": date_range}
    for i, var in enumerate(WEATHER_VARIABLES):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data).set_index("timestamp")
    df.to_parquet(out_path)
    logger.success(f"Saved weather → {out_path.name}")
    return df


def fetch_weather_forecast(city: str = "marrakech", days_ahead: int = 7) -> pd.DataFrame:
    """Fetch 7-day weather forecast from Open-Meteo for a given city."""
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Choose from {list(CITY_COORDS)}")

    lat, lon = CITY_COORDS[city]
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om_client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": WEATHER_VARIABLES,
        "forecast_days": days_ahead,
        "timezone": "UTC",
    }

    responses = om_client.weather_api(
        "https://api.open-meteo.com/v1/forecast", params=params
    )
    response = responses[0]
    hourly = response.Hourly()

    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": date_range}
    for i, var in enumerate(WEATHER_VARIABLES):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data).set_index("timestamp")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    combined = load_all_sources()
    print(combined.groupby("city").agg(
        rows=("load_kW", "count"),
        start=("load_kW", lambda s: s.index.min()),
        end=("load_kW", lambda s: s.index.max()),
        mean_kW=("load_kW", "mean"),
    ))

    # Fetch weather for UCI cities (Tetouan has weather embedded)
    for city in ["laayoune", "boujdour", "foum_eloued", "marrakech"]:
        fetch_openmeteo_weather(city)
