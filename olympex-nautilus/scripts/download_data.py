"""
Download and prepare BTC data from HuggingFace dataset for NautilusTrader backtesting.

Dataset: gionuibk/hyperliquidL2Book-v2
Data types:
  - data/candles/ : OHLCV candles (streaming snapshots, need dedup)
  - data/history_BTC_deep.parquet : Trade ticks (time, px, sz, side)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "gionuibk/hyperliquidL2Book-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_DIR = Path(__file__).parent.parent / "data"


def list_dataset_files(prefix: str = "") -> list[str]:
    """List files in the HuggingFace dataset."""
    files = list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    return sorted(files)


def download_file(filename: str) -> str:
    """Download a single file from the dataset."""
    return hf_hub_download(
        REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN,
    )


def download_btc_candles(max_files: int = 100) -> pd.DataFrame:
    """
    Download candle files and extract clean BTC 1-minute bars.

    The raw data contains streaming snapshots (multiple rows per timestamp).
    We take the last snapshot per timestamp to get finalized candles.
    """
    candle_files = list_dataset_files("data/candles/")
    print(f"Found {len(candle_files)} candle files, downloading up to {max_files}...")

    all_btc = []
    for i, fname in enumerate(candle_files[:max_files]):
        try:
            path = download_file(fname)
            df = pd.read_parquet(path)
            btc = df[df["coin"] == "BTC"].copy()
            if len(btc) > 0:
                all_btc.append(btc)
            if (i + 1) % 20 == 0:
                print(f"  Downloaded {i + 1}/{min(len(candle_files), max_files)} files...")
        except Exception as e:
            print(f"  Error downloading {fname}: {e}")
            continue

    if not all_btc:
        print("No BTC candle data found!")
        return pd.DataFrame()

    df = pd.concat(all_btc, ignore_index=True)
    print(f"Raw BTC rows: {len(df)}")

    # Deduplicate: keep last snapshot per timestamp (most complete candle)
    df = df.sort_values(["timestamp", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Deduplicated BTC bars: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def download_btc_trades() -> pd.DataFrame:
    """Download BTC trade tick data from history_BTC_deep.parquet."""
    print("Downloading BTC trade history...")
    path = download_file("data/history_BTC_deep.parquet")
    df = pd.read_parquet(path)

    # Convert epoch ms to datetime
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"BTC trades: {len(df)} ticks")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Price range: {df['px'].min():.2f} to {df['px'].max():.2f}")

    return df


def trades_to_candles(trades_df: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    """Convert trade ticks to OHLCV candles at given interval."""
    df = trades_df.set_index("datetime")

    candles = df["px"].resample(interval).ohlc()
    candles.columns = ["open", "high", "low", "close"]
    candles["volume"] = df["sz"].resample(interval).sum()
    candles["num_trades"] = df["px"].resample(interval).count()

    # Drop bars with no trades
    candles = candles.dropna(subset=["open"])
    candles = candles[candles["num_trades"] > 0]
    candles = candles.reset_index()
    candles.rename(columns={"datetime": "timestamp"}, inplace=True)

    print(f"Generated {len(candles)} candles at {interval} interval")
    return candles


def save_data(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame to parquet in data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved: {path} ({len(df)} rows)")
    return path


def main():
    """Download and prepare all BTC data."""
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable first")
        print("  export HF_TOKEN='your_huggingface_token'")
        sys.exit(1)

    # Strategy 1: Download candle files (OHLCV from exchange)
    print("=" * 60)
    print("DOWNLOADING BTC CANDLE DATA")
    print("=" * 60)
    candles_df = download_btc_candles(max_files=200)
    if len(candles_df) > 0:
        save_data(candles_df, "btc_candles_1min")
        print(f"\nCandle stats:")
        print(f"  Bars: {len(candles_df)}")
        print(f"  Time range: {candles_df['timestamp'].min()} to {candles_df['timestamp'].max()}")
        print(f"  Price range: {candles_df['close'].min():.2f} to {candles_df['close'].max():.2f}")

    # Strategy 2: Download trade ticks and convert to candles
    print("\n" + "=" * 60)
    print("DOWNLOADING BTC TRADE TICKS")
    print("=" * 60)
    trades_df = download_btc_trades()
    save_data(trades_df, "btc_trades_raw")

    # Convert trades to 1-min and 5-min candles
    candles_1m = trades_to_candles(trades_df, "1min")
    save_data(candles_1m, "btc_candles_from_trades_1min")

    candles_5m = trades_to_candles(trades_df, "5min")
    save_data(candles_5m, "btc_candles_from_trades_5min")

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    if len(candles_df) > 0:
        print(f"Exchange candles (1min): {len(candles_df)} bars")
    print(f"Trade-derived candles (1min): {len(candles_1m)} bars")
    print(f"Trade-derived candles (5min): {len(candles_5m)} bars")
    print(f"Raw trades: {len(trades_df)} ticks")
    print("\nData saved to:", DATA_DIR)


if __name__ == "__main__":
    main()
