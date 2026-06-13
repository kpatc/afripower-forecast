"""
Integration tests for the FastAPI service (api/main.py).
Uses httpx.TestClient — no server process required.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_status_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_has_timestamp():
    r = client.get("/health")
    assert "timestamp" in r.json()


def test_health_has_models_loaded():
    r = client.get("/health")
    assert "models_loaded" in r.json()


# ---------------------------------------------------------------------------
# /cities
# ---------------------------------------------------------------------------

def test_cities_returns_dict():
    r = client.get("/cities")
    assert r.status_code == 200
    assert "cities" in r.json()


def test_cities_contains_five():
    r = client.get("/cities")
    assert len(r.json()["cities"]) == 5


def test_cities_has_tetouan():
    r = client.get("/cities")
    assert "tetouan" in r.json()["cities"]


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------

def test_models_endpoint():
    r = client.get("/models")
    assert r.status_code == 200
    assert "available" in r.json()


# ---------------------------------------------------------------------------
# POST /forecast — input validation
# ---------------------------------------------------------------------------

def test_forecast_invalid_city():
    r = client.post("/forecast", json={"city": "casablanca", "model": "lightgbm", "horizon_days": 1})
    assert r.status_code == 422


def test_forecast_invalid_horizon_too_large():
    r = client.post("/forecast", json={"city": "tetouan", "model": "lightgbm", "horizon_days": 10})
    assert r.status_code == 422


def test_forecast_invalid_horizon_zero():
    r = client.post("/forecast", json={"city": "tetouan", "model": "lightgbm", "horizon_days": 0})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /forecast — happy path (requires trained models)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/lightgbm.pkl").exists(),
    reason="lightgbm.pkl not present — run make train",
)
def test_forecast_response_structure():
    r = client.post("/forecast", json={"city": "tetouan", "model": "lightgbm", "horizon_days": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["city"] == "tetouan"
    assert body["model"] == "lightgbm"
    assert body["horizon_days"] == 1
    assert body["n_points"] == 24
    assert len(body["forecasts"]) == 24


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/lightgbm.pkl").exists(),
    reason="lightgbm.pkl not present — run make train",
)
def test_forecast_kw_fields_present():
    r = client.post("/forecast", json={"city": "marrakech", "model": "lightgbm", "horizon_days": 1})
    assert r.status_code == 200
    first = r.json()["forecasts"][0]
    assert "forecast_kW" in first
    assert "lower_80_kW" in first
    assert "upper_80_kW" in first


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/lightgbm.pkl").exists(),
    reason="lightgbm.pkl not present — run make train",
)
def test_forecast_intervals_ordered():
    r = client.post("/forecast", json={"city": "tetouan", "model": "lightgbm", "horizon_days": 1})
    for point in r.json()["forecasts"]:
        assert point["lower_80_kW"] <= point["upper_80_kW"]


# ---------------------------------------------------------------------------
# GET /forecast/latest
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/xgboost.pkl").exists(),
    reason="xgboost.pkl not present — run make train",
)
def test_forecast_latest_get():
    r = client.get("/forecast/latest?city=laayoune&model=xgboost&horizon_days=2")
    assert r.status_code == 200
    assert r.json()["n_points"] == 48
