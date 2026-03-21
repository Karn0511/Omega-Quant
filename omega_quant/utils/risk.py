from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskState:
    """Class configuration auto-docstring."""
    daily_pnl: float = 0.0


@dataclass
class RiskManager:
    """Class configuration auto-docstring."""
    max_risk_per_trade: float
    max_daily_loss: float
    max_position_pct: float

    def can_trade(self, account_equity: float) -> bool:
        """Auto-docstring."""
        if account_equity <= 0:
            return False
        return True

    def daily_loss_breached(self, state: RiskState, account_equity: float) -> bool:
        """Auto-docstring."""
        if account_equity <= 0:
            return True
        return abs(min(state.daily_pnl, 0.0)) / account_equity >= self.max_daily_loss

    def position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        if entry_price <= 0 or stop_price <= 0:
            return 0.0

        risk_capital = account_equity * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_price)
        if price_risk == 0:
            return 0.0

        units = risk_capital / price_risk
        max_units = (account_equity * self.max_position_pct) / entry_price
        return max(0.0, min(units, max_units))
