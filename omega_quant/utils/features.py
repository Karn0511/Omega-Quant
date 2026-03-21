from __future__ import annotations

from typing import List, Tuple

# pylint: disable=import-error
import joblib
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler
# pylint: enable=import-error


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Auto-docstring."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Auto-docstring."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Auto-docstring."""
    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd = ema_fast - ema_slow
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Auto-docstring."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + (std * std_mult)
    lower = mid - (std * std_mult)
    return upper, mid, lower


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Auto-docstring."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Institutional-grade feature engineering with index-safe insertion."""
    out = df.copy()
    feature_dict = {}

    # Technical Indicators
    feature_dict["rsi"] = _rsi(out["close"], 14)
    
    macd, macd_signal, macd_hist = _macd(out["close"])
    feature_dict["macd"] = macd
    feature_dict["macd_signal"] = macd_signal
    feature_dict["macd_hist"] = macd_hist

    feature_dict["ema_20"] = _ema(out["close"], 20)
    feature_dict["ema_50"] = _ema(out["close"], 50)

    bb_upper, bb_mid, bb_lower = _bollinger(out["close"], 20)
    feature_dict["bb_upper"] = bb_upper
    feature_dict["bb_mid"] = bb_mid
    feature_dict["bb_lower"] = bb_lower

    vol_mean = out["volume"].rolling(20).mean()
    feature_dict["volume_spike"] = out["volume"] / vol_mean.replace(0, np.nan)

    # Secondary Metrics
    feature_dict["trend_strength"] = (feature_dict["ema_20"] - feature_dict["ema_50"]) / out["close"].replace(0, np.nan)

    returns = out["close"].pct_change()
    feature_dict["volatility_index"] = returns.rolling(30).std() * np.sqrt(1440)

    rolling_high = out["high"].rolling(50).max()
    rolling_low = out["low"].rolling(50).min()
    range_span = (rolling_high - rolling_low).replace(0, np.nan)
    feature_dict["liquidity_zone_pos"] = (out["close"] - rolling_low) / range_span

    feature_dict["atr"] = _atr(out, 14)

    # Safe Concatenation
    new_features = pd.DataFrame(feature_dict, index=out.index)
    out = pd.concat([out, new_features], axis=1)

    # Cleanup
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill().dropna().reset_index(drop=True)
    return out


def normalize_features(df: pd.DataFrame, feature_columns: List[str], scaler_path: str, fit: bool = True) -> pd.DataFrame:
    """Auto-docstring."""
    scaled_df = df.copy()

    if fit:
        scaler = StandardScaler()
        scaled_df[feature_columns] = scaler.fit_transform(scaled_df[feature_columns])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled_df[feature_columns] = scaler.transform(scaled_df[feature_columns])

    return scaled_df


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """Auto-docstring."""
    skip = {"timestamp", "datetime", "symbol", "target", "action"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
