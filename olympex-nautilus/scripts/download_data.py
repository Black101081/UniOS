"""
Download and prepare BTC 1-min candle data for NautilusTrader backtesting.

Two data sources, combined for maximum coverage:
  1. HuggingFace dataset (gionuibk/hyperliquidL2Book-v2) — historical candles
  2. Hyperliquid REST API — recent candles (last ~3.5 days)

Usage:
    export HF_TOKEN='your_token'
    python scripts/download_data.py
"""

import os
import sys

import pandas as pd
from pathlib import Path

REPO_ID = "gionuibk/hyperliquidL2Book-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_DIR = Path(__file__).parent.parent / "data"


def download_hf_candles() -> pd.DataFrame:
    """
    Download all candle files from HuggingFace and extract BTC 1-min bars.

    The raw data contains streaming snapshots (multiple rows per timestamp).
    We keep the last snapshot per timestamp to get finalized candles.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set, skipping HuggingFace download")
        return pd.DataFrame()

    files = sorted([
        f for f in list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
        if f.startswith("data/candles/")
    ])
    print(f"Found {len(files)} candle files on HuggingFace, downloading...")

    all_btc = []
    errors = 0

    for i, fname in enumerate(files):
        try:
            path = hf_hub_download(REPO_ID, fname, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_parquet(path)
            btc = df[df["coin"] == "BTC"].copy()
            if len(btc) > 0:
                all_btc.append(btc)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(files)}] {sum(len(x) for x in all_btc)} BTC rows so far")
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {fname}: {e}")
            continue

    print(f"Downloaded {len(files) - errors}/{len(files)} files ({errors} errors)")

    if not all_btc:
        print("No BTC candle data found in HuggingFace dataset!")
        return pd.DataFrame()

    df = pd.concat(all_btc, ignore_index=True)
    print(f"Raw BTC rows (with streaming duplicates): {len(df)}")

    # Deduplicate: keep last snapshot per timestamp (most complete candle)
    df = df.sort_values(["timestamp", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Deduplicated: {len(df)} unique 1-min candles")
    return df


def download_api_candles() -> pd.DataFrame:
    """Download recent candles from Hyperliquid REST API (no auth needed)."""
    from scripts.download_hyperliquid import download_candles

    print("Downloading recent candles from Hyperliquid API...")
    return download_candles(coin="BTC", days=5, interval="1m")


def merge_sources(hf_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    """Merge HuggingFace and API data, preferring API for overlapping timestamps."""
    dfs = [d for d in [hf_df, api_df] if len(d) > 0]
    if not dfs:
        return pd.DataFrame()

    common_cols = ["coin", "timestamp", "open", "high", "low", "close", "volume", "num_trades"]

    normalized = []
    for df in dfs:
        df = df.copy()
        # Normalize timestamps to timezone-naive UTC
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        normalized.append(df[common_cols])

    combined = pd.concat(normalized, ignore_index=True)
    # For overlapping timestamps, keep the row with higher volume (more complete)
    combined = combined.sort_values(["timestamp", "volume"])
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def main():
    """Download and prepare BTC candle data from all sources."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Source 1: HuggingFace historical candles
    print("=" * 60)
    print("SOURCE 1: HUGGINGFACE DATASET")
    print("=" * 60)
    hf_df = download_hf_candles()

    # Source 2: Hyperliquid API recent candles
    print("\n" + "=" * 60)
    print("SOURCE 2: HYPERLIQUID API (RECENT)")
    print("=" * 60)
    api_df = download_api_candles()

    # Merge
    print("\n" + "=" * 60)
    print("MERGING DATA SOURCES")
    print("=" * 60)
    combined = merge_sources(hf_df, api_df)

    if len(combined) == 0:
        print("ERROR: No data from any source!")
        sys.exit(1)

    # Save
    outpath = DATA_DIR / "btc_candles_1min.parquet"
    combined.to_parquet(outpath, index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("DATA SUMMARY")
    print(f"{'=' * 60}")
    print(f"HuggingFace candles: {len(hf_df) if len(hf_df) > 0 else 'N/A'}")
    print(f"API candles: {len(api_df) if len(api_df) > 0 else 'N/A'}")
    print(f"Combined (deduplicated): {len(combined)}")
    print(f"Time range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print(f"Price range: ${combined['close'].min():.2f} to ${combined['close'].max():.2f}")

    # Check continuity
    diffs = combined["timestamp"].diff().dt.total_seconds()
    gaps = diffs[diffs > 120]
    if len(gaps) > 0:
        total_gap_h = gaps.sum() / 3600
        print(f"Gaps > 2 min: {len(gaps)} (total gap time: {total_gap_h:.1f}h)")

    print(f"\nSaved to: {outpath}")


if __name__ == "__main__":
    main()
