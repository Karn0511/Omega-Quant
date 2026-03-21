from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd  # type: ignore


def load_symbol_parquet(parquet_dir: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Auto-docstring."""
    safe_symbol = symbol.replace("/", "_")
    path = Path(parquet_dir) / f"{safe_symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def latest_row(parquet_path: str | Path) -> Optional[pd.Series]:
    """Auto-docstring."""
    path = Path(parquet_path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return df.iloc[-1]
