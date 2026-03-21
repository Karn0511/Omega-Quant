from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


@dataclass
class SignalDecision:
    """Class configuration auto-docstring."""
    action: str
    confidence: float
    stop_loss: float
    take_profit: float
    trailing_stop: float


class StrategyEngine:
    """Class configuration auto-docstring."""
    def __init__(self, config: Dict):
        """Auto-docstring."""
        cfg = config["strategy"]
        self.buy_threshold = cfg["buy_threshold"]
        self.sell_threshold = cfg["sell_threshold"]
        self.stop_loss_mult = cfg["stop_loss_atr_mult"]
        self.take_profit_mult = cfg["take_profit_atr_mult"]
        self.trailing_mult = cfg["trailing_stop_atr_mult"]

    def from_probabilities(self, prob_sell: float, prob_hold: float, prob_buy: float, price: float, atr: float) -> SignalDecision:
        """Auto-docstring."""
        if prob_buy > self.buy_threshold:
            action = "BUY"
            confidence = prob_buy
            stop_loss = price - atr * self.stop_loss_mult
            take_profit = price + atr * self.take_profit_mult
            trailing_stop = price - atr * self.trailing_mult
        elif prob_buy < self.sell_threshold:
            action = "SELL"
            confidence = prob_sell
            stop_loss = price + atr * self.stop_loss_mult
            take_profit = price - atr * self.take_profit_mult
            trailing_stop = price + atr * self.trailing_mult
        else:
            action = "HOLD"
            confidence = prob_hold
            stop_loss = price
            take_profit = price
            trailing_stop = price

        return SignalDecision(action, float(confidence), float(stop_loss), float(take_profit), float(trailing_stop))

    def annotate_dataframe(self, df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
        """Auto-docstring."""
        out = df.copy()
        actions = []
        confidences = []
        stop_losses = []
        take_profits = []
        trailing_stops = []

        for i, row in out.iterrows():
            sell, hold, buy = probs[i]
            atr = float(row.get("atr", 0.0))
            atr = atr if atr > 0 else float(row["close"] * 0.002)
            decision = self.from_probabilities(sell, hold, buy, float(row["close"]), atr)
            actions.append(decision.action)
            confidences.append(decision.confidence)
            stop_losses.append(decision.stop_loss)
            take_profits.append(decision.take_profit)
            trailing_stops.append(decision.trailing_stop)

        out["action"] = actions
        out["confidence"] = confidences
        out["stop_loss"] = stop_losses
        out["take_profit"] = take_profits
        out["trailing_stop"] = trailing_stops
        out["signal"] = out["action"].map({"SELL": -1, "HOLD": 0, "BUY": 1})
        return out
