"""
Unit tests for src/evaluation/metrics.py
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import coverage, mae, mape, pinball_loss, rmse, evaluate_forecasts


@pytest.fixture()
def perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return y, y.copy()


@pytest.fixture()
def with_error():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 38.0, 52.0])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------

def test_mae_perfect(perfect):
    y, p = perfect
    assert mae(y, p) == pytest.approx(0.0)


def test_mae_known(with_error):
    y, p = with_error
    expected = np.mean(np.abs(y - p))
    assert mae(y, p) == pytest.approx(expected)


def test_mae_non_negative(with_error):
    assert mae(*with_error) >= 0


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

def test_rmse_perfect(perfect):
    y, p = perfect
    assert rmse(y, p) == pytest.approx(0.0)


def test_rmse_known(with_error):
    y, p = with_error
    expected = float(np.sqrt(np.mean((y - p) ** 2)))
    assert rmse(y, p) == pytest.approx(expected)


def test_rmse_ge_mae(with_error):
    assert rmse(*with_error) >= mae(*with_error)


# ---------------------------------------------------------------------------
# mape
# ---------------------------------------------------------------------------

def test_mape_perfect(perfect):
    assert mape(*perfect) == pytest.approx(0.0, abs=1e-6)


def test_mape_non_negative(with_error):
    assert mape(*with_error) >= 0


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_all_inside():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.5, 1.5, 2.5])
    hi = np.array([1.5, 2.5, 3.5])
    assert coverage(y, lo, hi) == pytest.approx(100.0)


def test_coverage_none_inside():
    y  = np.array([10.0, 20.0, 30.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert coverage(y, lo, hi) == pytest.approx(0.0)


def test_coverage_range():
    y  = np.linspace(0, 10, 100)
    lo = np.zeros(100)
    hi = np.ones(100) * 5
    val = coverage(y, lo, hi)
    assert 0 <= val <= 100


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------

def test_pinball_q50_is_half_mae():
    y    = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 2.0, 2.0])
    assert pinball_loss(y, pred, q=0.5) == pytest.approx(mae(y, pred) / 2, rel=1e-5)


def test_pinball_non_negative():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 1.5, 1.5])
    for q in [0.1, 0.5, 0.9]:
        assert pinball_loss(y, p, q) >= 0


# ---------------------------------------------------------------------------
# evaluate_forecasts
# ---------------------------------------------------------------------------

def test_evaluate_keys_no_intervals(with_error):
    metrics = evaluate_forecasts(*with_error)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE"}


def test_evaluate_keys_with_80(with_error):
    y, p = with_error
    metrics = evaluate_forecasts(y, p, lower_80=p - 2, upper_80=p + 2)
    assert "coverage_80" in metrics


def test_evaluate_values_finite(with_error):
    metrics = evaluate_forecasts(*with_error)
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} is not finite"
