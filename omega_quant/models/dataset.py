from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore


@dataclass
class DatasetArtifacts:
    """Class configuration auto-docstring."""
    x: np.ndarray
    y: np.ndarray
    feature_columns: List[str]


def build_action_labels(close: pd.Series, horizon: int = 1, threshold: float = 0.001) -> np.ndarray:
    """Auto-docstring."""
    future = close.shift(-horizon)
    ret = (future - close) / close.replace(0, np.nan)
    y = np.zeros(len(close), dtype=np.int64)
    y[ret > threshold] = 2
    y[ret < -threshold] = 0
    y[(ret >= -threshold) & (ret <= threshold)] = 1
    return y


def build_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    sequence_length: int,
    horizon: int,
    threshold: float = 0.001,
) -> DatasetArtifacts:
    labels = build_action_labels(df["close"], horizon=horizon, threshold=threshold)

    x, y = [], []
    features = df[feature_columns].to_numpy(dtype=np.float32)
    for i in range(sequence_length, len(df) - horizon):
        x.append(features[i - sequence_length : i])
        y.append(labels[i])

    return DatasetArtifacts(
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        feature_columns=feature_columns,
    )


class MarketSequenceDataset(Dataset):
    """Class configuration auto-docstring."""
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Auto-docstring."""
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        """Auto-docstring."""
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Auto-docstring."""
        return self.x[idx], self.y[idx]
