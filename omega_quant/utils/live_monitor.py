"""Phase 3: Real-Time Monitoring System."""
import logging
import time

LOGGER = logging.getLogger(__name__)

class LiveMonitor:
    """Live telemetry stream for alerts and risk exposure trends."""

    def __init__(self):
        """Auto-docstring."""
        self.last_pnl = 0.0
        self.last_trade_time = 0.0
        self.trade_count_last_hour = 0
        self.confidence_stream = []

    def update_metrics(self, equity: float, pnl: float, open_positions: int, confidence: float):
        """Auto-docstring."""
        self.confidence_stream.append(confidence)
        if len(self.confidence_stream) > 20:
            self.confidence_stream.pop(0)

        # Alert: Sudden Loss Spike
        if (pnl - self.last_pnl) < -500: # Example $500 sudden drop
            LOGGER.critical("MONITOR ALERT: Sudden PnL loss spike detected! ($%.2f)", pnl - self.last_pnl)

        self.last_pnl = pnl

        # Alert: Unusual Trade frequency
        now = time.time()
        if now - self.last_trade_time < 3600:
            self.trade_count_last_hour += 1
            if self.trade_count_last_hour > 20:
                LOGGER.warning("MONITOR ALERT: Hyperactive trading frequency (%d/hr).", self.trade_count_last_hour)
        else:
            self.trade_count_last_hour = 1
            self.last_trade_time = now

        # Alert: Confidence Collapse
        if len(self.confidence_stream) == 20:
            avg_conf = sum(self.confidence_stream) / 20.0
            if avg_conf < 0.45:
                LOGGER.warning("MONITOR ALERT: Global system confidence collapsing (Avg=%.2f).", avg_conf)

    def print_status(self, equity: float, daily_pnl: float, positions: int):
        """Auto-docstring."""
        LOGGER.info("[LIVE MONITOR] Equity: $%.2f | Daily PNL: $%.2f | Open Positions: %d",
                    equity, daily_pnl, positions)
