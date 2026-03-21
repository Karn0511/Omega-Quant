"""Phase 8: AI Strategy Switching (Market Regime Classifier)."""
import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

class StrategyManager:
    """Classifies market regime and switches between Trend, Mean-Reversion, Breakout."""

    def __init__(self, config: dict):
        """Auto-docstring."""
        self.config = config
        self.current_regime = "TREND_FOLLOWING"
        self.regimes = ["TREND_FOLLOWING", "MEAN_REVERSION", "BREAKOUT"]

    def classify_regime(self, df: pd.DataFrame) -> str:
        """Determines logic block based on volatility and ADX/trend alignment."""
        if len(df) < 50:
            return self.current_regime

        recent = df.iloc[-50:]
        close = recent["close"].values

        # Simple Regime Classification logic (ADX approximation via ATR + EMA slope)
        ema_20 = recent["close"].ewm(span=20).mean().values
        ema_50 = recent["close"].ewm(span=50).mean().values

        trend_strength = np.abs(ema_20[-1] - ema_50[-1]) / close[-1]
        volatility = np.std(np.diff(close) / close[:-1])

        if trend_strength > 0.015 and volatility < 0.02:
            self.current_regime = "TREND_FOLLOWING"
        elif volatility > 0.025 and trend_strength < 0.01:
            self.current_regime = "MEAN_REVERSION"
        elif volatility > 0.03 and trend_strength > 0.02:
            self.current_regime = "BREAKOUT"

        LOGGER.info("Market Regime Classifier: Evaluated %s", self.current_regime)
        return self.current_regime

    def apply_regime_modifiers(self, action: str, confidence: float) -> str:
        """Modifies action based on active Strategy."""
        if self.current_regime == "MEAN_REVERSION":
            # In Mean Reversion, extreme confidence might indicate blowout (contrarian)
            # OR we just tighten thresholds immensely
            if confidence < 0.8:
                return "HOLD"
        elif self.current_regime == "BREAKOUT":
            # In breakout, we don't hold if confidence is spiking
            if action == "HOLD" and confidence > 0.6:
                return "BUY" # Aggressive entry

        return action
