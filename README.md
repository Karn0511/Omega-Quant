# OMEGA-QUANT: QCNN Hybrid Trading Intelligence Engine

A fully local, modular AI trading system for crypto and stocks using a quantum-inspired + deep learning hybrid architecture.

## Features

- Historical OHLCV ingestion with CCXT (Binance) + yfinance (stocks)
- Real-time WebSocket market streaming
- Parquet storage + minute auto-updater
- Advanced feature engineering (RSI, MACD, EMA, Bollinger, trend, volatility, liquidity)
- Hybrid QCNN-inspired + CNN-LSTM model in PyTorch
- Training pipeline with tqdm + checkpointing + MB/s throughput
- Strategy engine with Buy/Sell/Hold probabilities and dynamic stops
- Risk controls (2% max risk/trade, max daily loss, position sizing)
- Backtesting with Backtrader (Profit %, Sharpe, Max Drawdown)
- Autonomous agent controller with periodic retraining

## Project Structure

```text
omega_quant/
  data/
  models/
  agents/
  strategies/
  backtesting/
  execution/
  utils/
  config/
  main.py
```

## Quick Start

1. Install dependencies:

```bash
pip install -r omega_quant/requirements.txt
```

1. Download historical data:

```bash
python -m omega_quant.main download
```

1. Train model:

```bash
python -m omega_quant.main train --symbol BTC/USDT
```

1. Backtest:

```bash
python -m omega_quant.main backtest --symbol BTC/USDT
```

1. Run autonomous agent:

```bash
python -m omega_quant.main agent
```

1. Optional safe live pipeline (Binance testnet by default):

```bash
python -m omega_quant.main trade-live --symbol BTC/USDT --amount 0.001
```

Use `--real` only when you intentionally want real orders.

Edit `omega_quant/config/config.yaml` to set `data.crypto_symbols` and `data.stock_symbols`.

## Safety Notes

- Start with paper trading only.
- Keep exchange keys out of source control.
- Validate strategy with out-of-sample and walk-forward tests before real capital.
