"""
AfriPower Forecast — FastAPI service.

Endpoints:
  GET  /health          → liveness check + loaded models
  GET  /cities          → available cities
  GET  /models          → available models
  POST /forecast        → J+1..J+7 forecast for a city (kW + 80% PI)
  GET  /forecast/latest → last forecast stored (GET-friendly demo)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"

CITY_LABELS = {
    "tetouan":     "Tétouan",
    "laayoune":    "Laâyoune",
    "boujdour":    "Boujdour",
    "foum_eloued": "Foum El Oued",
    "marrakech":   "Marrakech",
}

app = FastAPI(
    title="AfriPower Forecast API",
    description="Probabilistic hourly electricity demand forecasting — 5 Moroccan cities, J+1 to J+7.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Lazy loaders (cached for the lifetime of the process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    p = PROCESSED / "features.parquet"
    if not p.exists():
        raise RuntimeError("features.parquet not found — run `make train` first.")
    return pd.read_parquet(p)


@lru_cache(maxsize=1)
def _norm_stats() -> pd.DataFrame:
    p = PROCESSED / "norm_stats.csv"
    if not p.exists():
        raise RuntimeError("norm_stats.csv not found — run `make train` first.")
    return pd.read_csv(p, index_col=0)


@lru_cache(maxsize=4)
def _load_gbm(name: str) -> dict:
    p = PROCESSED / f"{name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{name}.pkl not found — run `make train`.")
    return joblib.load(p)


def _available_models() -> list[str]:
    names = []
    for m in ("lightgbm", "xgboost"):
        if (PROCESSED / f"{m}.pkl").exists():
            names.append(m)
    return names


def _denorm(values: np.ndarray, city: str) -> np.ndarray:
    stats = _norm_stats()
    mean = float(stats.loc[city, "mean_kW"])
    std  = float(stats.loc[city, "std_kW"])
    return values * std + mean


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ModelName = Literal["lightgbm", "xgboost"]


class ForecastRequest(BaseModel):
    city: Literal["tetouan", "laayoune", "boujdour", "foum_eloued", "marrakech"] = "tetouan"
    model: ModelName = "lightgbm"
    horizon_days: int = Field(default=7, ge=1, le=7)


class HourlyPoint(BaseModel):
    timestamp: str
    forecast_kW: float
    lower_80_kW: float
    upper_80_kW: float


class ForecastResponse(BaseModel):
    city: str
    city_label: str
    model: str
    horizon_days: int
    generated_at: str
    n_points: int
    forecasts: list[HourlyPoint]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    try:
        _features()
        _norm_stats()
        for m in _available_models():
            _load_gbm(m)
        logger.info(f"API ready. Models: {_available_models()}")
    except Exception as e:
        logger.warning(f"Startup warning: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": _available_models(),
        "features_ready": (PROCESSED / "features.parquet").exists(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/cities", tags=["meta"])
async def cities() -> dict:
    return {"cities": CITY_LABELS}


@app.get("/models", tags=["meta"])
async def models() -> dict:
    return {"available": _available_models()}


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
async def forecast(req: ForecastRequest) -> ForecastResponse:
    # Validate model availability
    available = _available_models()
    if req.model not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not trained yet. Available: {available}. Run `make train`.",
        )

    try:
        df    = _features()
        gbm   = _load_gbm(req.model)
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Filter to requested city and take the test slice (last 15%)
    city_col = f"city_{req.city}"
    if city_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"City column '{city_col}' not found in features.")

    city_df  = df[df[city_col] == 1].sort_index()
    n        = len(city_df)
    split    = int(n * 0.85)
    test_df  = city_df.iloc[split:]

    horizon_h = req.horizon_days * 24
    slice_df  = test_df.iloc[:horizon_h]

    if len(slice_df) == 0:
        raise HTTPException(status_code=500, detail="No test data available for this city.")

    TARGET = "load_norm"
    X = slice_df.drop(columns=[TARGET], errors="ignore")

    # Run quantile predictions
    fc_norm   = gbm[0.5].predict(X)
    lo_norm   = gbm[0.1].predict(X)
    hi_norm   = gbm[0.9].predict(X)

    # Denormalise to kW
    fc_kw = _denorm(fc_norm, req.city)
    lo_kw = _denorm(lo_norm, req.city)
    hi_kw = _denorm(hi_norm, req.city)

    forecasts = [
        HourlyPoint(
            timestamp=str(ts),
            forecast_kW=round(float(fc_kw[i]), 2),
            lower_80_kW=round(float(lo_kw[i]), 2),
            upper_80_kW=round(float(hi_kw[i]), 2),
        )
        for i, ts in enumerate(X.index)
    ]

    return ForecastResponse(
        city=req.city,
        city_label=CITY_LABELS[req.city],
        model=req.model,
        horizon_days=req.horizon_days,
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_points=len(forecasts),
        forecasts=forecasts,
    )


@app.get("/forecast/latest", response_model=ForecastResponse, tags=["forecast"])
async def forecast_latest(
    city: Literal["tetouan", "laayoune", "boujdour", "foum_eloued", "marrakech"] = "tetouan",
    model: ModelName = "lightgbm",
    horizon_days: int = 7,
) -> ForecastResponse:
    """GET-friendly version of /forecast for quick browser/curl testing."""
    return await forecast(ForecastRequest(city=city, model=model, horizon_days=horizon_days))
