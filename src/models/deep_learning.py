"""
LSTM-based sequence forecasting model (PyTorch).
Target: load_norm column, moved to position 0 before training.
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

TARGET = "load_norm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reorder(df: pd.DataFrame) -> pd.DataFrame:
    """Put load_norm first so target_idx=0 always holds."""
    cols = [TARGET] + [c for c in df.columns if c != TARGET]
    return df[cols]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, horizon: int):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon, 0]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Forecaster
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
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def _norm(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean_) / (self.std_ + 1e-8)

    def _denorm_target(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.std_[0] + 1e-8) + self.mean_[0]

    def fit(
        self,
        train: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> LSTMForecaster:
        df = _reorder(train)
        data = df.values.astype(np.float32)
        self.mean_ = data.mean(axis=0)
        self.std_  = data.std(axis=0)
        data = self._norm(data)

        dataset = TimeSeriesDataset(data, self.seq_len, self.horizon)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim   = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        logger.info(f"LSTM — training {epochs} epochs on {DEVICE} "
                    f"({len(dataset)} sequences) …")
        self.model.train()
        for epoch in range(1, epochs + 1):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optim.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optim.step()
                total += loss.item()
            if epoch % 5 == 0:
                logger.info(f"  epoch {epoch:3d}/{epochs}  loss={total/len(loader):.4f}")

        logger.success("LSTM training complete.")
        return self

    def predict(self, context: pd.DataFrame) -> np.ndarray:
        df = _reorder(context)
        data = self._norm(df.values.astype(np.float32))
        x = torch.tensor(data[-self.seq_len:][None], dtype=torch.float32).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()[0]
        return self._denorm_target(pred)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "mean": self.mean_,
            "std":  self.std_,
        }, path)
        logger.info(f"LSTM saved → {path}")

    @classmethod
    def load(cls, path: Path, input_size: int, horizon: int = 168) -> LSTMForecaster:
        obj = cls(input_size=input_size, horizon=horizon)
        ckpt = torch.load(path, map_location=DEVICE)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.mean_ = ckpt["mean"]
        obj.std_  = ckpt["std"]
        return obj
