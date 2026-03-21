"""Phase 7: Drawdown Defense System."""
import logging
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

class DrawdownDefense:
    """Monitors PNL and restricts size or halts trading during losing streaks."""

    def __init__(self, max_drawdown_pct: float = 0.15, max_losing_streak: int = 4):
        """Auto-docstring."""
        self.max_drawdown_pct = max_drawdown_pct
        self.max_losing_streak = max_losing_streak

        self.current_losing_streak = 0
        self.peak_equity = 10000.0
        self.current_equity = 10000.0
        self.paused = False

    def update_equity(self, equity: float, last_trade_profit: float):
        """Auto-docstring."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_losing_streak = 0
            self.paused = False
        else:
            if last_trade_profit < 0:
                self.current_losing_streak += 1
            else:
                self.current_losing_streak = 0

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown >= self.max_drawdown_pct or self.current_losing_streak >= self.max_losing_streak:
            self.paused = True
            LOGGER.warning("Drawdown Defense: TRADING PAUSED. DD=%.2f%%, Streak=%d", drawdown*100, self.current_losing_streak)
        else:
            self.paused = False

    def get_position_scaler(self) -> float:
        """Capital preservation mode: scale down position sizing if on a losing streak."""
        if self.paused:
            return 0.0

        if self.current_losing_streak >= self.max_losing_streak - 1:
            LOGGER.info("Drawdown Defense: Entering Capital Preservation Mode (0.2x sizing).")
            return 0.2
        elif self.current_losing_streak >= 2:
            return 0.5

        return 1.0
