from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import ccxt  # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf

LOGGER = logging.getLogger(__name__)


class MarketDataFetcher:
    """Class configuration auto-docstring."""
    def __init__(self, config: Dict):
        """Auto-docstring."""
        self.config = config
        self.exchange_name = config["data"]["exchange"]
        self.crypto_symbols = config["data"].get("crypto_symbols", [])
        self.stock_symbols = config["data"].get("stock_symbols", [])
        self.timeframe = config["data"]["timeframe"]
        self.limit = config["data"]["limit"]
        self.parquet_dir = Path(config["data"]["parquet_dir"])
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        exchange_cls = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_cls({"enableRateLimit": True})
        
        # Permanent Cloud Fix: Detect GitHub Actions and enable fallbacks immediately
        self.is_github = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
        if self.is_github:
            LOGGER.info("Permanent Cloud Fix: GitHub runner detected. Pre-engaging data fallbacks.")

    def _symbol_file(self, symbol: str) -> Path:
        """Auto-docstring."""
        safe_symbol = symbol.replace("/", "_")
        return self.parquet_dir / f"{safe_symbol}_{self.timeframe}.parquet"

    @staticmethod
    def _ohlcv_to_df(ohlcv: List[List], symbol: str) -> pd.DataFrame:
        """Auto-docstring."""
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["symbol"] = symbol
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"]).reset_index(drop=True)
        return df

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Auto-docstring."""
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.ffill().bfill().dropna().reset_index(drop=True)
        return df

    def fetch_historical(self, symbol: str, since_days: int) -> pd.DataFrame:
        """Auto-docstring."""
        # Permanent Cloud Fix: Force yfinance on GitHub to avoid any blocking delays
        if self.is_github:
            LOGGER.info("Cloud Mode: Using yfinance for %s (Binance block protection active)", symbol)
            ticker = symbol.replace("/", "-").replace("USDT", "USD")
            return self.fetch_stock_historical(ticker, since_days)

        since = datetime.now(timezone.utc) - timedelta(days=since_days)
        since_ms = int(since.timestamp() * 1000)

        all_rows: List[List] = []
        try:
            while True:
                batch = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since_ms, limit=self.limit)
                if not batch:
                    break
                all_rows.extend(batch)

                last_ts = batch[-1][0]
                next_since_ms = last_ts + 1
                if next_since_ms <= since_ms:
                    break
                since_ms = next_since_ms

                if len(batch) < self.limit:
                    break
                time.sleep(self.exchange.rateLimit / 1000)
        except Exception as exc:
            # Fallback for Restricted Locations (GitHub Actions)
            if "restricted location" in str(exc) or "451" in str(exc):
                LOGGER.warning("Restricted region detected! Falling back to yfinance for %s", symbol)
                ticker = symbol.replace("/", "-").replace("USDT", "USD")
                return self.fetch_stock_historical(ticker, since_days)
            raise exc

        if not all_rows:
            raise RuntimeError(f"No historical data fetched for {symbol}")

        df = self._ohlcv_to_df(all_rows, symbol)
        return self._clean(df)

    def fetch_stock_historical(self, ticker: str, since_days: int) -> pd.DataFrame:
        """Auto-docstring."""
        period_days = min(max(since_days, 1), 30)
        interval = "1m" if self.timeframe == "1m" else "5m"

        hist = yf.download(
            tickers=ticker,
            period=f"{period_days}d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if hist.empty:
            raise RuntimeError(f"No stock data fetched for {ticker}")

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0].lower() for c in hist.columns]
        else:
            hist.columns = [str(c).lower() for c in hist.columns]

        hist = hist.rename(columns={"adj close": "close"})
        needed = ["open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in hist.columns]
        if missing:
            raise ValueError(f"Stock data missing columns for {ticker}: {missing}")

        out = hist[needed].copy()
        out["timestamp"] = pd.to_datetime(out.index, utc=True)
        out["symbol"] = ticker
        out = out[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
        out = out.reset_index(drop=True)
        return self._clean(out)

    def save_symbol_data(self, symbol: str, df: pd.DataFrame) -> Path:
        """Auto-docstring."""
        target = self._symbol_file(symbol)
        if target.exists():
            existing = pd.read_parquet(target)
            merged = pd.concat([existing, df], ignore_index=True)
            merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"])
            merged = self._clean(merged)
        else:
            merged = df

        merged.to_parquet(target, index=False)
        LOGGER.info("Saved %s rows to %s", len(merged), target)
        return target

    def update_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Auto-docstring."""
        since_days = self.config["data"]["since_days"]
        results = {}
        
        for symbol in self.crypto_symbols:
            try:
                # Phase 7: Minimal Fetch Optimization
                df = self.fetch_historical(symbol, since_days=1) if os.getenv("GITHUB_ACTIONS") else self.fetch_historical(symbol, since_days)
                self.save_symbol_data(symbol, df)
                results[symbol] = df
            except Exception as exc: # pylint: disable=broad-exception-caught
                LOGGER.exception("Failed to update %s: %s", symbol, exc)

        for ticker in self.stock_symbols:
            if not ticker: continue
            try:
                df = self.fetch_stock_historical(ticker, since_days=1)
                self.save_symbol_data(ticker, df)
                results[ticker] = df
            except Exception as exc: # pylint: disable=broad-exception-caught
                LOGGER.exception("Failed to update %s: %s", ticker, exc)
                
        return results

    def auto_update_minutely(self) -> None:
        """Auto-docstring."""
        symbol_list = self.crypto_symbols + self.stock_symbols
        LOGGER.info("Starting minutely updater for symbols: %s", ", ".join(symbol_list))
        while True:
            started = time.time()
            self.update_all_symbols()
            elapsed = time.time() - started
            sleep_s = max(0.0, 60.0 - elapsed)
            time.sleep(sleep_s)
