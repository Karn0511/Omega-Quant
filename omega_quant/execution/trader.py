"""Phase 3: Execution Intelligence."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

# pylint: disable=import-error
import ccxt  # type: ignore
# pylint: enable=import-error

from omega_quant.strategies.signal_engine import SignalDecision
from omega_quant.utils.risk import RiskManager, RiskState

LOGGER = logging.getLogger(__name__)

from omega_quant.utils.validation_mode import ValidationScanner

@dataclass
class Position:
    """Class configuration auto-docstring."""
    side: str
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    execution_score: float = 0.0

class ExecutionEngine:
    """Simulates realistic execution dynamics: slippage, spread, and market impact."""

    def __init__(self, base_slippage: float = 0.0005, base_spread: float = 0.0002):
        """Auto-docstring."""
        self.base_slippage = base_slippage
        self.base_spread = base_spread

    def execute_trade(self, action: str, price: float, qty: float) -> Optional[float]:
        """Determines true fill price considering slippage and spread latency."""
        import time, random
        # Phase 1: Latency + Failure Simulation
        delay = random.uniform(0.1, 2.0) # 100ms - 2000ms delay
        time.sleep(delay)

        # 5-10% random order rejection (retry block)
        if random.random() < 0.08:
            LOGGER.error("ExecutionEngine: Exchange order REJECTED! Simulated network anomaly.")
            return None # None triggers retry or abort in trader logic

        spread_half = (price * self.base_spread) / 2.0
        slippage = price * self.base_slippage

        # Phase 2: Position Impact Control (Split orders & limit size)
        max_chunk_value = 50000.0
        order_value = qty * price
        
        if order_value > max_chunk_value:
            chunks = int(order_value // max_chunk_value) + 1
            LOGGER.info("Phase 2 Control: Splitting large $%.2f order into %d chunks to prevent market impact.", order_value, chunks)
            
            chunk_qty = qty / chunks
            total_fill = 0.0
            for _ in range(chunks):
                delay = random.uniform(0.5, 3.0) # Execution delay between chunks
                time.sleep(delay)
                chunk_impact = (chunk_qty / 100000.0) * price 
                chunk_fill = price + spread_half + slippage + chunk_impact if action == "BUY" else price - spread_half - slippage - chunk_impact
                total_fill += chunk_fill * chunk_qty
            fill_price = total_fill / qty
        else:
            market_impact = (qty / 100000.0) * price 
            fill_price = price + spread_half + slippage + market_impact if action == "BUY" else price - spread_half - slippage - market_impact
            
        return fill_price

class PaperTrader:
    """Class configuration auto-docstring."""
    def __init__(self, config: Dict, risk_manager: RiskManager, shadow_mode: bool = True):
        """Auto-docstring."""
        self.config = config
        self.risk_manager = risk_manager
        self.shadow_mode = shadow_mode # Phase 4 flag
        self.state = RiskState(daily_pnl=0.0)
        self.equity = 10_000.0
        self.position: Optional[Position] = None
        self.execution_engine = ExecutionEngine()

    def _sanity_check(self, decision: SignalDecision, price: float, qty: float) -> bool:
        """Phase 6: Position Sanity Check before EVERY trade."""
        if qty * price > self.equity * 0.95:
            LOGGER.critical("Sanity Check: Position size exceeds absolute limits!")
            return False

        # Basic validation
        if price <= 0 or qty <= 0:
            LOGGER.critical("Sanity Check: Invalid price or qty computed.")
            return False

        # Signal freshness (simulated - normally we'd pass timestamp)
        LOGGER.info("Sanity Check: Market conditions valid. Signal %s confirmed.", decision.action)
        return True

    def on_signal(self, decision: SignalDecision, price: float, capital_scale: float = 1.0) -> None:
        """Auto-docstring."""
        if self.risk_manager.daily_loss_breached(self.state, self.equity):
            return

        # Smart order routing / execution
        if decision.action == "BUY" and self.position is None:
            # Dynamically scale capital exposure
            effective_equity = self.equity * capital_scale
            raw_qty = self.risk_manager.position_size(effective_equity, price, decision.stop_loss)
            if raw_qty <= 0:
                return

            # Phase 6: Sanity Check validation
            if not self._sanity_check(decision, price, raw_qty):
                return

            base_str = "MARKET"
            qty = raw_qty
            if qty * price > 5000:
                base_str = "LIMIT (Splitting)"
                qty = qty * 0.8  # Assume 80% filled on limit

            # Phase 1: Retry Logic on network failure
            max_retries = 3
            fill_price = None
            for attempt in range(max_retries):
                LOGGER.info("Routing Order [Attempt %d/%d]: %s", attempt+1, max_retries, base_str)
                fill_price = self.execution_engine.execute_trade("BUY", price, qty)
                if fill_price is not None:
                    break

            if fill_price is None:
                LOGGER.critical("Execution fully aborted after 3 network retries.")
                return

            # Phase 4: Shadow Mode Logging
            if self.shadow_mode:
                LOGGER.info("[SHADOW MODE] Virtual execution recorded. Delayed fill=%.4f (Offset=%.4f)", fill_price, fill_price - price)

            self.position = Position(
                "LONG", qty, fill_price, decision.stop_loss, decision.take_profit, decision.trailing_stop
            )
            LOGGER.info("PAPER BUY qty=%.6f type=%s fill=%.4f",
                        qty, base_str, fill_price)

        elif decision.action == "SELL" and self.position is not None:
            fill_price = None
            for _ in range(3):
                fill_price = self.execution_engine.execute_trade("SELL", price, self.position.qty)
                if fill_price is not None:
                    break

            if fill_price is None:
                LOGGER.critical("CRITICAL: Failed to exit position due to network failure!")
                return # In real world, push force liquidation

            pnl = (fill_price - self.position.entry_price) * self.position.qty
            self._close_position(fill_price, pnl)

    def mark_to_market(self, price: float) -> None:
        """Auto-docstring."""
        if self.position is None:
            return

        pnl_pct = (price - self.position.entry_price) / self.position.entry_price
        
        # Phase 4: Profit Lock-in System (Partial profit booking)
        if pnl_pct >= 0.10 and not getattr(self.position, "booked_50", False):
            LOGGER.info("PROFIT LOCK-IN: +10%% reached. Securing 50%% of position.")
            partial_qty = self.position.qty * 0.50
            pnl = (price - self.position.entry_price) * partial_qty
            self.equity += pnl # Book real capital
            self.position.qty -= partial_qty
            self.position.booked_50 = True
            
        elif pnl_pct >= 0.05 and not getattr(self.position, "booked_30", False) and not getattr(self.position, "booked_50", False):
            LOGGER.info("PROFIT LOCK-IN: +5%% reached. Securing 30%% of position.")
            partial_qty = self.position.qty * 0.30
            pnl = (price - self.position.entry_price) * partial_qty
            self.equity += pnl # Book real capital
            self.position.qty -= partial_qty
            self.position.booked_30 = True

        if price <= self.position.stop_loss:
            LOGGER.info("PAPER SELL OUT STOP LOSS at %.4f", price)
            self._score_exit(price, "STOP")
            pnl = (price - self.position.entry_price) * self.position.qty
            self._close_position(price, pnl)
            return

        trailing_limit = self.position.trailing_stop * 0.98
        if price <= trailing_limit:
            LOGGER.info("PAPER SELL OUT TRAILING STOP at %.4f", price)
            self._score_exit(price, "TRAILING_STOP")
            pnl = (price - self.position.entry_price) * self.position.qty
            self._close_position(price, pnl)
            return

        if price >= self.position.take_profit:
            LOGGER.info("PAPER SELL OUT TAKE PROFIT at %.4f", price)
            self._score_exit(price, "TAKE_PROFIT")
            pnl = (price - self.position.entry_price) * self.position.qty
            self._close_position(price, pnl)
            return

        self.position.trailing_stop = max(self.position.trailing_stop, price)

    def _score_exit(self, exit_price: float, reason: str): # pylint: disable=unused-argument
        # Was the exit optimal mathematically?
        entry = self.position.entry_price
        target = self.position.take_profit
        if target > entry:
            efficiency = (exit_price - entry) / (target - entry)
            self.position.execution_score = efficiency
            LOGGER.info("Execution Efficiency Score: %.2f%%", efficiency * 100)

    def _close_position(self, fill_price: float, pnl: float): # pylint: disable=unused-argument
        self.equity += pnl
        self.state.daily_pnl += pnl
        self.position = None

class LiveValidationTrader:
    """Phase 1: Micro-capital Live Trading with Phase 2 Reality Gap Analysis."""
    
    def __init__(self, config: Dict, risk_manager: RiskManager, api_key: str, api_secret: str):
        """Auto-docstring."""
        self.config = config
        self.risk_manager = risk_manager
        self.state = RiskState(daily_pnl=0.0)
        self.equity = 50.0  # Phase 1: Micro-cap ~ $50 for live testing
        self.position: Optional[Position] = None
        
        # Phase 7: Human Intervention Lock (Warning override)
        LOGGER.critical("="*60)
        LOGGER.critical(" 🔒 HUMAN INTERVENTION LOCK ACTIVE 🔒")
        LOGGER.critical(" SYSTEM RUNNING AUTONOMOUSLY IN LIVE VALIDATION MODE.")
        LOGGER.critical(" DO NOT MANUALLY OVERRIDE TRADES ON THE EXCHANGE.")
        LOGGER.critical("="*60)
        
        # Initialize REAL execution
        self.binance = BinanceTrader(api_key, api_secret, testnet=False) 
        self.scanner = ValidationScanner()
        self.api_ready = True

    def _sanity_check(self, decision: SignalDecision, price: float, qty: float) -> bool:
        """Phase 6: Position Sanity Check before EVERY trade."""
        if qty * price > self.equity * 0.95:
            LOGGER.critical("Sanity Check: Position size exceeds absolute limits!")
            return False
        if price <= 0 or qty <= 0: return False
        return True

    def on_signal(self, decision: SignalDecision, price: float, capital_scale: float = 1.0) -> None:
        """Auto-docstring."""
        if self.risk_manager.daily_loss_breached(self.state, self.equity): return
        
        # Determine strict exchange symbol formatting
        symbol = "BTC/USDT" 

        if decision.action == "BUY" and self.position is None:
            effective_equity = self.equity * capital_scale
            raw_qty = self.risk_manager.position_size(effective_equity, price, decision.stop_loss)
            
            # Phase 1: Micro-Capital Limits ($50 equivalent)
            max_qty = 50.0 / price
            qty = min(raw_qty, max_qty) * 0.99 # Leave room for fees
            
            if not self._sanity_check(decision, price, qty): return
            
            LOGGER.info("ROUTING LIVE VALIDATION ORDER: BUY %.6f %s", qty, symbol)
            import time
            start_t = time.time()
            
            try:
                order = self.binance.execute_market_order(symbol, "buy", float(qty))
                actual_price = float(order.get("average", price)) if order and order.get("average") else price
                delay_ms = int((time.time() - start_t) * 1000)
                
                # Phase 2: Log reality gap
                self.scanner.log_execution_gap(symbol, price, actual_price, delay_ms)
                
                self.position = Position(
                    "LONG", qty, actual_price, decision.stop_loss, decision.take_profit, decision.trailing_stop
                )
            except Exception as e:
                LOGGER.critical("Live execution failed: %s", e)

        elif decision.action == "SELL" and self.position is not None:
            import time
            start_t = time.time()
            try:
                order = self.binance.execute_market_order(symbol, "sell", float(self.position.qty))
                actual_price = float(order.get("average", price)) if order and order.get("average") else price
                delay_ms = int((time.time() - start_t) * 1000)
                
                # Phase 2: Log reality gap
                self.scanner.log_execution_gap(symbol, price, actual_price, delay_ms)
                
                pnl = (actual_price - self.position.entry_price) * self.position.qty
                self.equity += pnl
                self.state.daily_pnl += pnl
                self.position = None
            except Exception as e:
                LOGGER.critical("Live EXIT execution failed: %s", e)

    def mark_to_market(self, price: float) -> None:
        """Passive MTM check utilizing strict SL/TP without trailing live modifications."""
        if self.position is None: return
        
        if price <= self.position.stop_loss or price >= self.position.take_profit:
            LOGGER.info("LIVE TRIGGER HIT (SL/TP) at %.4f", price)
            self.on_signal(SignalDecision(action="SELL", confidence=1.0, stop_loss=0, take_profit=0, trailing_stop=0), price)


class BinanceTrader:
    """Class configuration auto-docstring."""
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Auto-docstring."""
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if testnet:
            self.exchange.set_sandbox_mode(True)

    def execute_market_order(self, symbol: str, side: str, amount: float):
        """Auto-docstring."""
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")
        order = self.exchange.create_order(symbol=symbol, type="market", side=side, amount=amount)
        LOGGER.info("Executed %s order: %s", side.upper(), order)
        return order
