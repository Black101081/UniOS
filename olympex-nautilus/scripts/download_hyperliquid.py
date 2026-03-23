"""
Download real BTC 1-minute candles directly from Hyperliquid REST API.

No authentication required. Uses the public candleSnapshot endpoint.

Usage:
    python scripts/download_hyperliquid.py [--days 30] [--coin BTC]
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = Path(__file__).parent.parent / "data"
MAX_CANDLES_PER_REQUEST = 5000


def fetch_candles(
    coin: str = "BTC",
    interval: str = "1m",
    start_ms: int = 0,
    end_ms: int = 0,
) -> list[dict]:
    """Fetch candles from Hyperliquid candleSnapshot endpoint."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_candles(coin: str = "BTC", days: int = 30, interval: str = "1m") -> pd.DataFrame:
    """
    Download candles for the given number of days at the specified interval.

    The API returns max 5000 candles per request.
    API data availability varies by interval:
      - 1m: ~3.5 days
      - 5m: ~17 days
      - 15m/1h: ~30 days
    """
    # Interval in minutes for pagination
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    mins = interval_minutes.get(interval, 1)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    cursor_start = start_ms
    request_count = 0

    print(f"Downloading {coin} {interval} candles for {days} days...")
    print(f"  From: {datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)}")
    print(f"  To:   {datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)}")

    while cursor_start < end_ms:
        # Each request covers up to 5000 candles worth of time
        cursor_end = min(cursor_start + MAX_CANDLES_PER_REQUEST * mins * 60 * 1000, end_ms)

        try:
            candles = fetch_candles(coin, interval, cursor_start, cursor_end)
            request_count += 1

            if not candles:
                # No data in this window, skip forward
                cursor_start = cursor_end
                continue

            all_candles.extend(candles)

            if request_count % 5 == 0:
                last_ts = datetime.fromtimestamp(candles[-1]["t"] / 1000, tz=timezone.utc)
                print(f"  [{request_count}] Fetched {len(all_candles)} candles so far... (last: {last_ts})")

            # Move cursor past the last candle we got
            cursor_start = candles[-1]["t"] + mins * 60_000

            # Rate limiting: be polite to the API
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}, retrying in 2s...")
            time.sleep(2)
            continue

    print(f"  Total API requests: {request_count}")
    print(f"  Total raw candles: {len(all_candles)}")

    if not all_candles:
        print("No candles received!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_candles)

    # API fields: t (open ms), T (close ms), s (coin), i (interval),
    #             o (open), h (high), l (low), c (close), v (volume), n (num_trades)
    df = df.rename(columns={
        "t": "open_time_ms",
        "T": "close_time_ms",
        "s": "coin",
        "i": "interval",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "num_trades",
    })

    # Convert string prices/volumes to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Create proper timestamp column
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)

    # Deduplicate by timestamp (keep last in case of overlaps)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Select final columns matching the format expected by run_backtest.py
    result = df[["coin", "timestamp", "open", "high", "low", "close", "volume", "num_trades"]].copy()

    print(f"\nFinal dataset:")
    print(f"  Candles: {len(result)}")
    print(f"  Time range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"  Price range: {result['close'].min():.2f} to {result['close'].max():.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Download Hyperliquid BTC candles")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download (default: 30)")
    parser.add_argument("--coin", type=str, default="BTC", help="Coin symbol (default: BTC)")
    parser.add_argument("--interval", type=str, default="1m",
                        choices=["1m", "5m", "15m", "1h"],
                        help="Candle interval (default: 1m). Longer intervals go further back.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = download_candles(coin=args.coin, days=args.days, interval=args.interval)

    if len(df) == 0:
        print("No data downloaded. Exiting.")
        return

    # Save with interval in filename
    interval_label = args.interval.replace("m", "min").replace("h", "hr")
    outpath = DATA_DIR / f"{args.coin.lower()}_candles_{interval_label}.parquet"
    df.to_parquet(outpath, index=False)
    print(f"\nSaved to: {outpath}")


if __name__ == "__main__":
    main()
