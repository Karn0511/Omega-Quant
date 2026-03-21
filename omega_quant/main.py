from __future__ import annotations

import argparse
import logging
import os
from typing import cast

# pylint: disable=import-error
import pandas as pd  # type: ignore
# pylint: enable=import-error

from omega_quant.agents.controller import OmegaAgentController
from omega_quant.backtesting.bt_engine import run_backtest
from omega_quant.data.fetcher import MarketDataFetcher
from omega_quant.data.websocket_stream import run_stream
from omega_quant.execution.trader import BinanceTrader
from omega_quant.models.pipeline import (
    predict_probabilities,
    prepare_training_data,
    train_model,
    load_model,
)
from omega_quant.strategies.signal_engine import StrategyEngine
from omega_quant.utils.config_loader import load_config
from omega_quant.utils.features import build_features
from omega_quant.utils.io import load_symbol_parquet
from omega_quant.utils.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)


def _default_symbol(config: dict) -> str:
    """Auto-docstring."""
    crypto = config["data"].get("crypto_symbols", [])
    stocks = config["data"].get("stock_symbols", [])
    if crypto:
        return crypto[0]
    if stocks:
        return stocks[0]
    raise ValueError("No symbols configured in data.crypto_symbols or data.stock_symbols")


def cmd_download(config: dict, symbol: str | None, timeframe: str | None) -> None:
    """Auto-docstring."""
    fetcher = MarketDataFetcher(config)
    if timeframe:
        fetcher.timeframe = timeframe

    if symbol:
        if "/" in symbol:
            df = fetcher.fetch_historical(symbol, config["data"]["since_days"])
        else:
            df = fetcher.fetch_stock_historical(symbol, config["data"]["since_days"])
        fetcher.save_symbol_data(symbol, df)
    else:
        fetcher.update_all_symbols()


def cmd_auto_update(config: dict) -> None:
    """Auto-docstring."""
    fetcher = MarketDataFetcher(config)
    fetcher.auto_update_minutely()


def cmd_stream(config: dict, symbol: str | None) -> None:
    """Auto-docstring."""
    symbol = symbol or _default_symbol(config)
    run_stream(
        symbol=symbol,
        timeframe=config["data"]["timeframe"],
        output_dir=config["data"]["parquet_dir"],
        reconnect_seconds=config["data"]["stream_reconnect_seconds"],
    )


def cmd_train(config: dict, symbol: str | None) -> None:
    """Auto-docstring."""
    symbol = symbol or _default_symbol(config)
    df = load_symbol_parquet(config["data"]["parquet_dir"], symbol, config["data"]["timeframe"])
    max_rows = int(config["training"].get("max_rows", len(df)))
    df = df.tail(max_rows).reset_index(drop=True)
    train_model(df, config)


def cmd_backtest(config: dict, symbol: str | None) -> None:
    """Auto-docstring."""
    symbol = symbol or _default_symbol(config)
    df = load_symbol_parquet(config["data"]["parquet_dir"], symbol, config["data"]["timeframe"])
    max_rows = int(config["training"].get("max_rows", len(df)))
    df = df.tail(max_rows).reset_index(drop=True)

    artifacts = prepare_training_data(df, config, fit_scaler=False)
    model = load_model(config, input_dim=len(artifacts.feature_columns))

    probs = predict_probabilities(model, artifacts.x)

    seq_len = config["features"]["sequence_length"]
    horizon = config["features"]["prediction_horizon"]
    feature_df = build_features(df)
    start_idx = seq_len
    end_idx = len(feature_df) - horizon
    fdf = cast(pd.DataFrame, feature_df.iloc[start_idx:end_idx, :].reset_index(drop=True))

    strategy = StrategyEngine(config)
    signals = strategy.annotate_dataframe(fdf, probs)
    result = run_backtest(signals, config)

    LOGGER.info(
        "Backtest results: profit=%.2f%% sharpe=%.4f max_drawdown=%.2f%%",
        result.profit_pct,
        result.sharpe,
        result.max_drawdown,
    )


def cmd_trade_once(config: dict) -> None:
    """Executes a single institutional trading pass (Serverless mode)."""
    from omega_quant.agents.controller import OmegaAgentController
    OmegaAgentController(config).run_once()


def cmd_trade_live(
    config: dict,
    symbol: str | None,
    amount: float,
    api_key: str | None,
    api_secret: str | None,
    testnet: bool,
) -> None:
    symbol = symbol or _default_symbol(config)
    api_key = api_key or os.getenv("BINANCE_API_KEY")
    api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Binance API credentials are required via args or BINANCE_API_KEY/BINANCE_API_SECRET")

    df = load_symbol_parquet(config["data"]["parquet_dir"], symbol, config["data"]["timeframe"])
    max_rows = int(config["training"].get("max_rows", len(df)))
    df = df.tail(max_rows).reset_index(drop=True)

    artifacts = prepare_training_data(df, config, fit_scaler=False)
    model = load_model(config, input_dim=len(artifacts.feature_columns))

    probs = predict_probabilities(model, artifacts.x[-1:])[0]
    feature_df = build_features(df)
    row = feature_df.iloc[-1]

    strategy = StrategyEngine(config)
    decision = strategy.from_probabilities(
        prob_sell=float(probs[0]),
        prob_hold=float(probs[1]),
        prob_buy=float(probs[2]),
        price=float(row["close"]),
        atr=float(row.get("atr", row["close"] * 0.002)),
    )
    LOGGER.info("Live trade decision: %s confidence=%.3f", decision.action, decision.confidence)

    trader = BinanceTrader(api_key=api_key, api_secret=api_secret, testnet=testnet)
    if decision.action == "BUY":
        trader.execute_market_order(symbol=symbol, side="buy", amount=amount)
    elif decision.action == "SELL":
        trader.execute_market_order(symbol=symbol, side="sell", amount=amount)
    else:
        LOGGER.info("Decision is HOLD, no live order submitted.")


def cmd_agent(config: dict) -> None:
    """Auto-docstring."""
    OmegaAgentController(config).run()


def build_parser() -> argparse.ArgumentParser:
    """Auto-docstring."""
    parser = argparse.ArgumentParser(description="OMEGA-QUANT: QCNN Hybrid Trading Intelligence Engine")
    parser.add_argument("--config", default="omega_quant/config/config.yaml", help="Path to YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download historical OHLCV data")
    download_parser.add_argument("--symbol", default=None)
    download_parser.add_argument("--timeframe", default=None)
    subparsers.add_parser("auto-update", help="Auto update OHLCV data every minute")

    stream_parser = subparsers.add_parser("stream", help="Start Binance WebSocket kline stream")
    stream_parser.add_argument("--symbol", default=None)

    train_parser = subparsers.add_parser("train", help="Train QCNN hybrid model")
    train_parser.add_argument("--symbol", default=None)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting with Backtrader")
    backtest_parser.add_argument("--symbol", default=None)

    trade_parser = subparsers.add_parser("trade-once", help="Generate one trade decision from latest market state")
    trade_parser.add_argument("--symbol", default=None)

    live_trade_parser = subparsers.add_parser("trade-live", help="Submit optional live/testnet Binance order")
    live_trade_parser.add_argument("--symbol", default=None)
    live_trade_parser.add_argument("--amount", type=float, default=0.001)
    live_trade_parser.add_argument("--api-key", default=None)
    live_trade_parser.add_argument("--api-secret", default=None)
    live_trade_parser.add_argument("--real", action="store_true", help="Submit to real Binance instead of testnet")

    subparsers.add_parser("agent", help="Run autonomous AI trading controller")

    return parser


def main() -> None:
    """Auto-docstring."""
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["app"]["log_level"])

    command_map = {
        "download": lambda: cmd_download(config, args.symbol, args.timeframe),
        "auto-update": lambda: cmd_auto_update(config),
        "stream": lambda: cmd_stream(config, args.symbol),
        "train": lambda: cmd_train(config, args.symbol),
        "backtest": lambda: cmd_backtest(config, args.symbol),
        "trade-once": lambda: cmd_trade_once(config),
        "trade-live": lambda: cmd_trade_live(
            config,
            args.symbol,
            args.amount,
            args.api_key,
            args.api_secret,
            not args.real,
        ),
        "agent": lambda: cmd_agent(config),
    }

    command_map[args.command]()


if __name__ == "__main__":
    main()
