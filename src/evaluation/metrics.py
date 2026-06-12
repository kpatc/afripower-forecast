"""
Forecasting evaluation metrics: MAE, RMSE, MAPE, and coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)) * 100)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def evaluate_forecasts(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    lower_80: pd.Series | np.ndarray | None = None,
    upper_80: pd.Series | np.ndarray | None = None,
    lower_95: pd.Series | np.ndarray | None = None,
    upper_95: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

    if lower_80 is not None and upper_80 is not None:
        metrics["coverage_80"] = coverage(y_true, np.asarray(lower_80), np.asarray(upper_80))

    if lower_95 is not None and upper_95 is not None:
        metrics["coverage_95"] = coverage(y_true, np.asarray(lower_95), np.asarray(upper_95))

    return metrics


def walk_forward_cv(
    df: pd.DataFrame,
    model_factory,
    n_folds: int = 5,
    horizon: int = 168,
    target_col: str = "load_MW",
) -> pd.DataFrame:
    """Expanding-window cross-validation."""
    results = []
    fold_size = (len(df) - horizon) // n_folds

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        train = df.iloc[:train_end]
        test = df.iloc[train_end : train_end + horizon]

        if len(test) < horizon:
            break

        model = model_factory()
        model.fit(train)

        X_test = test.drop(columns=[target_col])
        preds = model.predict(X_test)
        metrics = evaluate_forecasts(test[target_col].values, preds["forecast"].values)
        metrics["fold"] = fold
        results.append(metrics)

    return pd.DataFrame(results)
