"""
FastAPI service exposing a /forecast endpoint.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

app = FastAPI(
    title="AfriPower Forecast API",
    description="J+1 to J+7 electricity demand forecast for Morocco",
    version="1.0.0",
)

# Lazy-loaded model registry
_MODELS: dict = {}


def _load_models() -> None:
    global _MODELS
    if _MODELS:
        return

    from src.models.ml_models import LightGBMForecaster

    lgbm_path = ROOT / "data" / "processed" / "lightgbm.pkl"
    if lgbm_path.exists():
        _MODELS["lightgbm"] = LightGBMForecaster.load(lgbm_path)
        logger.info("LightGBM model loaded.")
    else:
        logger.warning("No trained model found — run `make train` first.")


@app.on_event("startup")
async def startup_event() -> None:
    _load_models()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    horizon_days: int = 7
    model: Literal["lightgbm", "xgboost", "lstm"] = "lightgbm"
    include_weather: bool = True


class HourlyForecast(BaseModel):
    timestamp: str
    forecast_MW: float
    lower_80_MW: float
    upper_80_MW: float


class ForecastResponse(BaseModel):
    model: str
    generated_at: str
    horizon_days: int
    forecasts: list[HourlyForecast]
    metrics_last_eval: dict | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models_loaded": list(_MODELS.keys())}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    _load_models()

    if req.model not in _MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not available. Loaded: {list(_MODELS.keys())}",
        )

    model = _MODELS[req.model]
    horizon_h = req.horizon_days * 24

    try:
        from src.data.fetch_data import fetch_weather_forecast
        from src.data.preprocess import build_features, clean_load

        # Build a minimal feature DataFrame for the forecast window using weather
        weather = fetch_weather_forecast(days_ahead=req.horizon_days + 1) if req.include_weather else None

        # Create a dummy load series (last known value repeated) to generate features
        now = pd.Timestamp.now(tz="UTC").floor("1h")
        future_index = pd.date_range(now, periods=horizon_h, freq="1h", tz="UTC")
        dummy_load = pd.Series(np.nan, index=future_index, name="load_MW")

        df = build_features(dummy_load, weather)
        df = df.drop(columns=["load_MW"], errors="ignore")
        df = df.iloc[:horizon_h]

        preds = model.predict(df)

    except Exception as exc:
        logger.error(f"Forecast error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    forecasts = [
        HourlyForecast(
            timestamp=str(ts),
            forecast_MW=round(float(row["forecast"]), 1),
            lower_80_MW=round(float(row["lower_80"]), 1),
            upper_80_MW=round(float(row["upper_80"]), 1),
        )
        for ts, row in preds.iterrows()
    ]

    return ForecastResponse(
        model=req.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizon_days=req.horizon_days,
        forecasts=forecasts,
    )


@app.get("/models")
async def list_models() -> dict:
    return {"available_models": list(_MODELS.keys())}
