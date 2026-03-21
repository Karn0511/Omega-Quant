# pylint: disable=import-error
import os
import logging
import numpy as np  # type: ignore
import ccxt  # type: ignore
# pylint: enable=import-error

LOGGER = logging.getLogger(__name__)

class LiquidityIntelligence:
    """Live Order Book scans finding imbalances, whale walls, and voids."""
    def __init__(self, exchange_id: str = "binance"):
        """Auto-docstring."""
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.is_github = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
        if self.is_github:
            LOGGER.info("Liquidity Cloud Fix: Order book scanning permanently disabled for serverless stability.")

    def fetch_order_book(self, symbol: str, limit: int = 100):
        """Auto-docstring."""
        if self.is_github:
            return None
            
        try:
            return self.exchange.fetch_order_book(symbol.replace("/", ""), limit)
        except Exception as exc:
            # Fallback for standard symbol or restricted region
            if "restricted location" in str(exc) or "451" in str(exc):
                LOGGER.warning("Liquidity Intelligence: Restricted region detected. Skipping order book scan for %s", symbol)
                return None
            try:
                return self.exchange.fetch_order_book(symbol, limit)
            except Exception as final_exc: # pylint: disable=broad-exception-caught
                LOGGER.error("Liquidity: Failed level2 data for %s: %s", symbol, final_exc)
                return None

    def analyze_liquidity(self, symbol: str) -> dict:
        """Computes short-term momentum using raw depth matrix."""
        ob = self.fetch_order_book(symbol)
        if not ob:
            return {"imbalance": 0.0, "whale_buy": 0.0, "whale_sell": 0.0, "liquidity_void": 0.0}

        bids = np.array(ob["bids"]) # [price, amount]
        asks = np.array(ob["asks"])

        if len(bids) == 0 or len(asks) == 0:
            return {"imbalance": 0.0, "whale_buy": 0.0, "whale_sell": 0.0, "liquidity_void": 0.0}

        # 1. Order Book Imbalance (-1 to +1 scale)
        total_bid_vol = np.sum(bids[:, 1])
        total_ask_vol = np.sum(asks[:, 1])
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-9)

        # 2. Identify Large Whale Orders (Volume Clusters)
        bid_mean, bid_std = np.mean(bids[:, 1]), np.std(bids[:, 1])
        ask_mean, ask_std = np.mean(asks[:, 1]), np.std(asks[:, 1])

        # Clusters defined as > 3 Std Devs above the local resting volume
        whale_bids = bids[bids[:, 1] > bid_mean + 3 * bid_std]
        whale_asks = asks[asks[:, 1] > ask_mean + 3 * ask_std]

        whale_buy_pressure = np.sum(whale_bids[:, 1]) if len(whale_bids) > 0 else 0
        whale_sell_pressure = np.sum(whale_asks[:, 1]) if len(whale_asks) > 0 else 0

        # 3. Track Sudden Liquidity Voids (Low density per $1 price move)
        bid_density = total_bid_vol / (bids[0, 0] - bids[-1, 0] + 1e-9)
        ask_density = total_ask_vol / (asks[-1, 0] - asks[0, 0] + 1e-9)
        liquidity_void = 1.0 / (bid_density + ask_density + 1e-9)

        metrics = {
            "imbalance": imbalance,
            "whale_buy": whale_buy_pressure,
            "whale_sell": whale_sell_pressure,
            "liquidity_void": liquidity_void
        }

        LOGGER.info("[%s LIQUIDITY] Imbalance: %.2f | Whale Buy: %.2f | Whale Sell: %.2f | Void: %.4f",
                    symbol, imbalance, whale_buy_pressure, whale_sell_pressure, liquidity_void)

        return metrics
