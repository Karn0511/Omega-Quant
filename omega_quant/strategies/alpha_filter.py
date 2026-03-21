"""Phase 2: Alpha Filter Layer."""
import logging
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

class AlphaFilter:
    """Class configuration auto-docstring."""
    def __init__(self, config: dict):
        """Auto-docstring."""
        self.config = config

    def evaluate_signal(self, action: str, confidence: float, row: Dict[str, Any], liq_metrics: Dict[str, float] = None) -> bool:
        """Only allow trades when ALL alpha filters pass."""
        if action == "HOLD":
            return True

        liq = liq_metrics or {}
        imbalance = liq.get("imbalance", 0.0)
        whale_buy = liq.get("whale_buy", 0.0)
        whale_sell = liq.get("whale_sell", 0.0)

        # 0. Liquidity Intelligence Front-run checking
        if action == "BUY":
            if imbalance < -0.3: # Massive Sell-side dominance
                LOGGER.warning("Alpha: BUY Reject. book sell imbalance (%.2f).", imbalance)
                return False
            if whale_sell > whale_buy * 3 and whale_sell > 0:
                LOGGER.warning("Alpha: BUY Reject. WHALE SELL WALL ahead.")
                return False

        elif action == "SELL":
            if imbalance > 0.3: # Massive Buy-side dominance
                LOGGER.warning("Alpha: SELL Reject. book buy imbalance (%.2f).", imbalance)
                return False
            if whale_buy > whale_sell * 3 and whale_buy > 0:
                LOGGER.warning("Alpha: SELL Reject. WHALE BUY WALL ahead.")
                return False

        # 1. Dynamic Confidence Threshold via Alpha Decomposition Edge
        import json
        from pathlib import Path
        decomp_path = Path("omega_quant/data/alpha_decomposition.json")
        base_threshold = 0.65
        conf_max = 1.0
        
        if decomp_path.exists():
            try:
                with open(decomp_path, "r", encoding="utf-8") as f:
                    ds = json.load(f)
                    band = ds.get("best_band", [0.65, 1.0])
                    base_threshold, conf_max = float(band[0]), float(band[1])
            except Exception:
                pass

        volatility = row.get("volatility_index", 0.0)
        dynamic_threshold = base_threshold + (volatility * 0.1)

        if confidence < dynamic_threshold or confidence > conf_max:
            LOGGER.debug("Alpha Filter: Rejected due to optimal confidence band bounds (%.2f not in %.2f - %.2f)", confidence, dynamic_threshold, conf_max)
            return False

        # 2. Market Volatility Filter
        atr = row.get("atr", 0.0)
        close = row.get("close", 1.0)
        atr_pct = atr / close
        if atr_pct > 0.05: # Extreme volatility block
            LOGGER.debug("Alpha Filter: Rejected due to extreme volatility (%.4f)", atr_pct)
            return False

        # 3. Volume Confirmation
        volume = row.get("volume", 0.0)
        volume_ma = row.get("volume_ma_20", volume)
        if volume < volume_ma * 0.8:
            LOGGER.debug("Alpha Filter: Rejected due to low volume confirmation")
            return False

        # 4. Trend Alignment
        ema_20 = row.get("ema_20", close)
        ema_50 = row.get("ema_50", close)
        trend_up = ema_20 > ema_50

        if action == "BUY" and not trend_up:
            LOGGER.debug("Alpha Filter: BUY Rejected against downward trend")
            return False
        if action == "SELL" and trend_up:
            LOGGER.debug("Alpha Filter: SELL Rejected against upward trend")
            return False

        LOGGER.info("Alpha Filter: Signal APPROVED (%s, conf=%.2f, order_book=%.2f)", action, confidence, list(liq.values())[0] if liq else 0)
        return True
