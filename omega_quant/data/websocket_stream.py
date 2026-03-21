from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd  # type: ignore
import websockets

LOGGER = logging.getLogger(__name__)


class BinanceKlineStreamer:
    """Class configuration auto-docstring."""
    def __init__(self, symbol: str, timeframe: str, output_dir: str, reconnect_seconds: int = 5):
        """Auto-docstring."""
        self.symbol = symbol
        self.timeframe = timeframe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reconnect_seconds = reconnect_seconds

    def _stream_url(self) -> str:
        """Auto-docstring."""
        normalized = self.symbol.replace("/", "").lower()
        return f"wss://stream.binance.com:9443/ws/{normalized}@kline_{self.timeframe}"

    def _output_file(self) -> Path:
        """Auto-docstring."""
        safe_symbol = self.symbol.replace("/", "_")
        return self.output_dir / f"stream_{safe_symbol}_{self.timeframe}.parquet"

    def _append_row(self, row: Dict) -> None:
        """Auto-docstring."""
        path = self._output_file()
        frame = pd.DataFrame([row])
        if path.exists():
            existing = pd.read_parquet(path)
            out = pd.concat([existing, frame], ignore_index=True)
            out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"])
        else:
            out = frame
        out = out.ffill().bfill()
        out.to_parquet(path, index=False)

    async def run(self) -> None:
        while True:
            try:
                LOGGER.info("Connecting to Binance stream: %s", self._stream_url())
                async with websockets.connect(self._stream_url(), ping_interval=20, ping_timeout=20) as ws:
                    async for raw in ws:
                        payload = json.loads(raw)
                        kline = payload.get("k", {})
                        row = {
                            "timestamp": datetime.fromtimestamp(kline.get("t", 0) / 1000, tz=timezone.utc),
                            "open": float(kline.get("o", 0.0)),
                            "high": float(kline.get("h", 0.0)),
                            "low": float(kline.get("l", 0.0)),
                            "close": float(kline.get("c", 0.0)),
                            "volume": float(kline.get("v", 0.0)),
                            "symbol": self.symbol,
                            "is_closed": bool(kline.get("x", False)),
                        }
                        self._append_row(row)
            except BaseException as exc: # pylint: disable=broad-exception-caught
                LOGGER.exception("Stream dropped for %s: %s", self.symbol, exc)
                await asyncio.sleep(self.reconnect_seconds)


def run_stream(symbol: str, timeframe: str, output_dir: str, reconnect_seconds: Optional[int] = 5) -> None:
    """Auto-docstring."""
    streamer = BinanceKlineStreamer(symbol, timeframe, output_dir, reconnect_seconds or 5)
    asyncio.run(streamer.run())
