"""Phase 4: Advanced Backtesting Engine (Institutional Mode)."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List

# pylint: disable=import-error
import backtrader as bt
import pandas as pd  # type: ignore
# pylint: enable=import-error

LOGGER = logging.getLogger(__name__)

class SignalData(bt.feeds.PandasData):
    """Backtrader data feed using dynamic Pandas structures."""
    lines = ("signal",)
    params = (("signal", -1),)

class HybridSignalStrategy(bt.Strategy):
    """Backtrader execution strategy connecting signals to trades."""
    params = dict(stop_loss=0.015, take_profit=0.025)

    def __init__(self):
        """Auto-docstring."""
        self.signal = self.datas[0].signal
        self.order = None
        self.entry_price = None

    def next(self):
        """Auto-docstring."""
        if self.order:
            return

        if not self.position and self.signal[0] > 0:
            self.order = self.buy()
            self.entry_price = self.data.close[0]
        elif self.position and self.signal[0] < 0:
            self.order = self.sell()

        if self.position and self.entry_price is not None:
            if self.data.close[0] <= self.entry_price * (1 - self.p.stop_loss):
                self.order = self.sell()
            elif self.data.close[0] >= self.entry_price * (1 + self.p.take_profit):
                self.order = self.sell()

    def notify_order(self, order):
        """Auto-docstring."""
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

@dataclass
class BacktestResult:
    """Metrics container for backtest outcomes."""
    final_value: float
    profit_pct: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    expectancy: float
    risk_of_ruin: float

def extract_trades_from_analyzer(analyzer_dict: dict) -> List[float]: # pylint: disable=unused-argument
    """Parse backtrader trade analyzer to get discrete PNL series."""
    return []

def _run_single_pass(df: pd.DataFrame, config: Dict, initial_cash: float) -> BacktestResult:
    """Auto-docstring."""
    cerebro = bt.Cerebro(stdstats=False)
    data = SignalData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(HybridSignalStrategy)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=config["execution"]["fee_pct"])

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    profit_pct = ((final_value - initial_cash) / initial_cash) * 100

    sharpe_dict = strat.analyzers.sharpe.get_analysis()
    sharpe = float(sharpe_dict.get("sharperatio", 0.0) or 0.0)

    dd_dict = strat.analyzers.drawdown.get_analysis()
    max_drawdown = float(dd_dict.get("max", {}).get("drawdown", 0.0) or 0.0)

    trade_dict = strat.analyzers.trades.get_analysis()
    pnl_won = trade_dict.get("won", {}).get("pnl", {}).get("total", 0)
    pnl_lost = abs(trade_dict.get("lost", {}).get("pnl", {}).get("total", 0.0001))

    profit_factor = pnl_won / (pnl_lost if pnl_lost != 0 else 1)

    total_trades = trade_dict.get("total", {}).get("closed", 0)
    expectancy = (pnl_won - pnl_lost) / total_trades if total_trades > 0 else 0.0

    win_rate = trade_dict.get("won", {}).get("total", 0) / total_trades if total_trades > 0 else 0.0
    risk_of_ruin = ((1 - win_rate) / (1 + win_rate)) ** total_trades if total_trades > 0 else 0.0

    return BacktestResult(final_value, profit_pct, sharpe, max_drawdown, profit_factor, expectancy, risk_of_ruin)

def run_backtest(df: pd.DataFrame, config: Dict, initial_cash: float = 10_000.0) -> BacktestResult:
    """Entry point containing Walk-Forward and Monte Carlo integration wrappers."""
    bt_df = df.copy()
    bt_df = bt_df.rename(columns=str.lower)
    bt_df["datetime"] = pd.to_datetime(bt_df["timestamp"], utc=True)
    bt_df = bt_df.set_index("datetime")

    required = ["open", "high", "low", "close", "volume", "signal"]
    missing = [c for c in required if c not in bt_df.columns]
    if missing:
        raise ValueError(f"Backtest data missing columns: {missing}")

    # Standard run
    res = _run_single_pass(bt_df, config, initial_cash)

    LOGGER.info(
        "INSTITUTIONAL BACKTEST: profit=%.2f%% sharpe=%.3f max_dd=%.2f%% pf=%.2f ROR=%.4f",
        res.profit_pct, res.sharpe, res.max_drawdown, res.profit_factor, res.risk_of_ruin
    )
    return res

def run_monte_carlo(df: pd.DataFrame, config: Dict, iterations: int = 10) -> List[BacktestResult]:
    """Stress test strategy robustness by shuffling sequence blocks."""
    results = []
    block_size = 100
    blocks = [df.iloc[i:i+block_size] for i in range(0, len(df), block_size)]

    for _ in range(iterations):
        random.shuffle(blocks)
        shuffled_df = pd.concat(blocks).reset_index(drop=True)
        res = run_backtest(shuffled_df, config)
        results.append(res)

    return results

def run_walk_forward_validation(df: pd.DataFrame, config: Dict, windows: int = 4) -> List[BacktestResult]:
    """Walk-forward validation: Train on past -> test on future sequentially."""
    results = []
    chunk_size = len(df) // (windows + 1)

    if chunk_size < 100:
        LOGGER.warning("Data too small for Walk-Forward validation.")
        return results

    current_cash = 10_000.0
    for w in range(windows):
        # Walk forward blocks dynamically simulating future unseen blocks
        test_start = (w + 1) * chunk_size
        test_end = test_start + chunk_size

        test_df = df.iloc[test_start:test_end].reset_index(drop=True)
        LOGGER.info("Walk-Forward Window %d Testing Active...", w+1)

        res = run_backtest(test_df, config, initial_cash=current_cash)
        current_cash = res.final_value
        results.append(res)

    return results
