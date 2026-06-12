"""
LSTM-based sequence forecasting model using PyTorch.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        horizon: int,
        target_idx: int = 0,
    ):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_idx = target_idx

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 168,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMForecaster:
    def __init__(self, input_size: int, horizon: int = 168):
        cfg = CFG["models"]["lstm"]
        self.seq_len = cfg["sequence_length"]
        self.horizon = horizon
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            horizon=horizon,
        ).to(DEVICE)
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.scaler_mean) / (self.scaler_std + 1e-8)

    def _denormalize_target(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.scaler_std[0] + 1e-8) + self.scaler_mean[0]

    def fit(
        self,
        train: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> LSTMForecaster:
        data = train.values.astype(np.float32)
        self.scaler_mean = data.mean(axis=0)
        self.scaler_std = data.std(axis=0)
        data = self._normalize(data)

        dataset = TimeSeriesDataset(data, self.seq_len, self.horizon)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        logger.info(f"Training LSTM for {epochs} epochs on {DEVICE} …")
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            if epoch % 5 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}  loss={avg:.4f}")

        logger.success("LSTM training complete.")
        return self

    def predict(self, context: pd.DataFrame) -> np.ndarray:
        data = context.values.astype(np.float32)
        data = self._normalize(data)
        x = torch.tensor(data[-self.seq_len :][None], dtype=torch.float32).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()[0]
        return self._denormalize_target(pred)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }, path)
        logger.info(f"LSTM model saved → {path}")

    @classmethod
    def load(cls, path: Path, input_size: int, horizon: int = 168) -> LSTMForecaster:
        obj = cls(input_size=input_size, horizon=horizon)
        ckpt = torch.load(path, map_location=DEVICE)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.scaler_mean = ckpt["scaler_mean"]
        obj.scaler_std = ckpt["scaler_std"]
        return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.fetch_data import fetch_entsoe_load, fetch_openmeteo_weather
    from src.data.preprocess import build_features, train_test_split_temporal
    from src.evaluation.metrics import evaluate_forecasts

    load = fetch_entsoe_load()
    weather = fetch_openmeteo_weather()
    df = build_features(load, weather)
    train_df, test_df = train_test_split_temporal(df)

    horizon = CFG["data"]["forecast_horizon"] * 24

    with mlflow.start_run(run_name="lstm"):
        forecaster = LSTMForecaster(input_size=train_df.shape[1], horizon=horizon)
        forecaster.fit(train_df, epochs=30)
        preds_raw = forecaster.predict(test_df)
        preds = pd.Series(preds_raw, index=test_df.index[:horizon])
        metrics = evaluate_forecasts(test_df["load_MW"].iloc[:horizon], preds)
        mlflow.log_metrics(metrics)
        mlflow.log_params(CFG["models"]["lstm"])
        forecaster.save(ROOT / "data" / "processed" / "lstm.pt")
