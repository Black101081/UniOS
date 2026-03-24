"""
Download BTC/USDT 1-minute kline data from Binance public API.
Paginates through all data from 2023-01-01 to 2026-03-24.
Saves as parquet file.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone

# Config
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 24, 23, 59, 59, tzinfo=timezone.utc)
OUTPUT_PATH = "/home/user/UniOS/olympex-nautilus/data/btc_binance_1min.parquet"
ENDPOINT = "https://api.binance.com/api/v3/klines"
RATE_LIMIT_SLEEP = 0.5
MAX_RETRIES = 3

start_ms = int(START_DATE.timestamp() * 1000)
end_ms = int(END_DATE.timestamp() * 1000)

all_candles = []
current_start = start_ms
total_fetched = 0
request_count = 0

print(f"Downloading {SYMBOL} {INTERVAL} candles from Binance")
print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
print(f"Expected ~1.6M+ candles")
print()

t0 = time.time()

while current_start < end_ms:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": current_start,
        "endTime": end_ms,
        "limit": LIMIT,
    }

    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            success = True
            break
        except Exception as e:
            print(f"  Retry {attempt}/{MAX_RETRIES} - Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    if not success:
        print(f"FAILED after {MAX_RETRIES} retries at startTime={current_start}. Stopping.")
        break

    if not data:
        print("No more data returned. Done.")
        break

    request_count += 1

    for row in data:
        all_candles.append({
            "coin": "BTC",
            "timestamp": pd.Timestamp(row[0], unit="ms", tz="UTC"),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "num_trades": int(row[8]),
        })

    total_fetched += len(data)

    if total_fetched % 50000 < LIMIT:
        elapsed = time.time() - t0
        rate = total_fetched / elapsed if elapsed > 0 else 0
        print(f"  {total_fetched:>10,} candles fetched | {request_count} requests | {elapsed:.0f}s | {rate:.0f} candles/s")

    # Advance past the last candle's open_time
    current_start = data[-1][0] + 60_000  # +1 minute in ms

    time.sleep(RATE_LIMIT_SLEEP)

elapsed = time.time() - t0
print()
print(f"Download complete: {total_fetched:,} candles in {elapsed:.0f}s ({request_count} requests)")

if all_candles:
    df = pd.DataFrame(all_candles)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"File size: {pd.io.common.file_exists(OUTPUT_PATH)}")
else:
    print("No data downloaded.")
