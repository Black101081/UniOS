"""
Hyperliquid WebSocket client for real-time BTC candle and trade data.

Connects to wss://api.hyperliquid.xyz/ws, subscribes to BTC 1m candles
and trades, prints data to stdout, and saves candles to a growing parquet
file every 100 candles.

Handles reconnection with exponential backoff on disconnect.

Usage:
    python scripts/hyperliquid_ws.py [--output data/btc_ws_candles.parquet]
    python scripts/hyperliquid_ws.py --test   # self-test with mock data

Press Ctrl+C to stop.
"""

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websocket

WS_URL = "wss://api.hyperliquid.xyz/ws"
SAVE_EVERY = 100  # Save parquet every N new candles


class HyperliquidWSClient:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.candles: list[dict] = []
        self.trade_count = 0
        self.candle_count = 0
        self.unsaved_candle_count = 0
        self.connected = False
        self.running = True
        self.ws: websocket.WebSocketApp | None = None
        self.reconnect_delay = 1.0  # seconds, grows with backoff
        self.max_reconnect_delay = 60.0
        self._lock = threading.Lock()

        # Load existing parquet if present
        if self.output_path.exists():
            try:
                existing = pd.read_parquet(self.output_path)
                self.candles = existing.to_dict("records")
                self.candle_count = len(self.candles)
                self._log(f"Loaded {self.candle_count} existing candles from {self.output_path}")
            except Exception as e:
                self._log(f"Could not load existing parquet: {e}")

    @staticmethod
    def _log(msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def _status(self):
        state = "CONNECTED" if self.connected else "DISCONNECTED"
        self._log(f"Status: {state} | candles={self.candle_count} | trades={self.trade_count}")

    # -- WebSocket callbacks --

    def _on_open(self, ws):
        self.connected = True
        self.reconnect_delay = 1.0  # reset backoff on successful connect
        self._log(f"Connected to {WS_URL}")

        # Subscribe to BTC 1m candles
        candle_sub = {
            "method": "subscribe",
            "subscription": {"type": "candle", "coin": "BTC", "interval": "1m"},
        }
        ws.send(json.dumps(candle_sub))
        self._log("Subscribed to BTC 1m candles")

        # Subscribe to BTC trades
        trade_sub = {
            "method": "subscribe",
            "subscription": {"type": "trades", "coin": "BTC"},
        }
        ws.send(json.dumps(trade_sub))
        self._log("Subscribed to BTC trades")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            self._log(f"Bad JSON: {message[:200]}")
            return

        channel = data.get("channel")

        if channel == "candle":
            self._handle_candle(data.get("data", {}))
        elif channel == "trades":
            self._handle_trades(data.get("data", []))
        elif channel == "subscriptionResponse":
            self._log(f"Subscription confirmed: {data}")
        else:
            # Log unknown messages briefly
            preview = json.dumps(data)[:200]
            self._log(f"MSG ({channel}): {preview}")

    def _on_error(self, ws, error):
        self._log(f"WS error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        self._log(f"Disconnected (code={close_status_code}, msg={close_msg})")

    # -- Data handlers --

    def _handle_candle(self, candle_data: dict):
        """Process a candle update."""
        # Hyperliquid candle fields: t, T, s, i, o, h, l, c, v, n
        record = {
            "coin": candle_data.get("s", "BTC"),
            "timestamp": pd.Timestamp(candle_data.get("t", 0), unit="ms", tz="UTC"),
            "open": float(candle_data.get("o", 0)),
            "high": float(candle_data.get("h", 0)),
            "low": float(candle_data.get("l", 0)),
            "close": float(candle_data.get("c", 0)),
            "volume": float(candle_data.get("v", 0)),
            "num_trades": int(candle_data.get("n", 0)),
        }

        with self._lock:
            # Deduplicate: replace candle with same timestamp, or append
            replaced = False
            for i, c in enumerate(self.candles):
                if c["timestamp"] == record["timestamp"]:
                    self.candles[i] = record
                    replaced = True
                    break
            if not replaced:
                self.candles.append(record)
                self.candle_count += 1
                self.unsaved_candle_count += 1

        action = "UPD" if replaced else "NEW"
        self._log(
            f"CANDLE [{action}] {record['timestamp']} "
            f"O={record['open']:.1f} H={record['high']:.1f} "
            f"L={record['low']:.1f} C={record['close']:.1f} "
            f"V={record['volume']:.2f} (total={self.candle_count})"
        )

        # Periodic save
        if self.unsaved_candle_count >= SAVE_EVERY:
            self._save_parquet()

    def _handle_trades(self, trades: list):
        """Process trade updates."""
        if not isinstance(trades, list):
            trades = [trades]
        for trade in trades:
            self.trade_count += 1
            coin = trade.get("coin", "BTC")
            px = trade.get("px", "?")
            sz = trade.get("sz", "?")
            side = trade.get("side", "?")
            ts = trade.get("time", 0)
            t_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3] if ts else "?"
            if self.trade_count <= 5 or self.trade_count % 50 == 0:
                self._log(
                    f"TRADE  {coin} {side:>1} px={px} sz={sz} @ {t_str} "
                    f"(total={self.trade_count})"
                )

    # -- Persistence --

    def _save_parquet(self):
        with self._lock:
            if not self.candles:
                return
            df = pd.DataFrame(self.candles)

        # Ensure proper types
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)
        self.unsaved_candle_count = 0
        self._log(f"Saved {len(df)} candles to {self.output_path}")

    # -- Main loop --

    def run(self):
        """Run the WebSocket client with reconnection logic."""
        self._log("Starting Hyperliquid WS client...")
        self._log(f"Output: {self.output_path}")

        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # run_forever blocks until disconnected
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                self._log(f"Connection exception: {e}")

            if not self.running:
                break

            self._log(f"Reconnecting in {self.reconnect_delay:.1f}s...")
            time.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    def stop(self):
        """Gracefully stop the client."""
        self._log("Shutting down...")
        self.running = False
        if self.ws:
            self.ws.close()
        # Final save
        self._save_parquet()
        self._status()
        self._log("Done.")


def run_self_test():
    """Self-test: exercise all data-handling logic with synthetic messages."""
    import tempfile

    print("=" * 60)
    print("SELF-TEST: verifying candle/trade handling + parquet save")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_candles.parquet"
        client = HyperliquidWSClient(out)
        client.unsaved_candle_count = 0

        base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Simulate 105 candle messages (triggers save at 100)
        for i in range(105):
            candle_msg = {
                "channel": "candle",
                "data": {
                    "t": base_ts + i * 60_000,
                    "T": base_ts + (i + 1) * 60_000 - 1,
                    "s": "BTC",
                    "i": "1m",
                    "o": str(87000.0 + i),
                    "h": str(87100.0 + i),
                    "l": str(86900.0 + i),
                    "c": str(87050.0 + i),
                    "v": str(1.5 + i * 0.01),
                    "n": 42 + i,
                },
            }
            client._on_message(None, json.dumps(candle_msg))

        assert client.candle_count == 105, f"Expected 105 candles, got {client.candle_count}"
        assert out.exists(), "Parquet file should exist after 100 candles"

        # Verify parquet content
        df = pd.read_parquet(out)
        assert len(df) >= 100, f"Expected >=100 rows in parquet, got {len(df)}"
        print(f"  Parquet has {len(df)} rows, columns: {list(df.columns)}")

        # Simulate candle update (same timestamp = replace, not duplicate)
        update_msg = {
            "channel": "candle",
            "data": {
                "t": base_ts,  # same as first candle
                "T": base_ts + 59_999,
                "s": "BTC", "i": "1m",
                "o": "87000.0", "h": "87200.0", "l": "86800.0",
                "c": "87100.0", "v": "2.5", "n": 99,
            },
        }
        client._on_message(None, json.dumps(update_msg))
        assert client.candle_count == 105, "Update should not increase count"

        # Simulate trades
        for i in range(5):
            trade_msg = {
                "channel": "trades",
                "data": [
                    {"coin": "BTC", "px": "87050.0", "sz": "0.5",
                     "side": "B", "time": base_ts + i * 1000},
                ],
            }
            client._on_message(None, json.dumps(trade_msg))

        assert client.trade_count == 5, f"Expected 5 trades, got {client.trade_count}"

        # Final save
        client._save_parquet()
        df2 = pd.read_parquet(out)
        assert len(df2) == 105, f"Expected 105 final rows, got {len(df2)}"

        # Verify reload from existing parquet
        client2 = HyperliquidWSClient(out)
        assert client2.candle_count == 105, f"Reload: expected 105, got {client2.candle_count}"

        client._status()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Hyperliquid WS client for BTC candles + trades")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "btc_ws_candles.parquet"),
        help="Output parquet path (default: data/btc_ws_candles.parquet)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run self-test with mock data instead of connecting to WS",
    )
    args = parser.parse_args()

    if args.test:
        run_self_test()
        return

    output_path = Path(args.output)
    client = HyperliquidWSClient(output_path)

    # Handle Ctrl+C gracefully
    def _sigint_handler(sig, frame):
        client.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    client.run()


if __name__ == "__main__":
    main()
