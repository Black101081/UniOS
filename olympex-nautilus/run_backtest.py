"""
Olympex Backtest Runner — Run adaptive strategy on real BTC data.

Usage:
    python run_backtest.py [--data-source candles|trades] [--interval 1min|5min]
"""

import sys
import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler

from strategies.olympex_strategy import OlympexStrategy, OlympexStrategyConfig

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def create_btcusdt_instrument() -> CryptoPerpetual:
    """Create a BTCUSDT perpetual instrument for Hyperliquid."""
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.HYPERLIQUID"),
        raw_symbol=Symbol("BTCUSDT-PERP"),
        base_currency=Currency.from_str("BTC"),
        quote_currency=Currency.from_str("USDT"),
        settlement_currency=Currency.from_str("USDT"),
        is_inverse=False,
        price_precision=1,
        size_precision=5,
        price_increment=Price.from_str("0.1"),
        size_increment=Quantity.from_str("0.00001"),
        max_quantity=Quantity.from_str("1000.00000"),
        min_quantity=Quantity.from_str("0.00001"),
        max_notional=None,
        min_notional=Money(10.00, Currency.from_str("USDT")),
        max_price=Price.from_str("1000000.0"),
        min_price=Price.from_str("0.1"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
        ts_event=0,
        ts_init=0,
    )


def load_candle_data(instrument: CryptoPerpetual, source: str = "candles", interval: str = "1min") -> list[Bar]:
    """Load candle data and convert to NautilusTrader Bar objects."""

    if source == "candles":
        filepath = DATA_DIR / "btc_candles_1min.parquet"
    elif source == "trades":
        filepath = DATA_DIR / f"btc_candles_from_trades_{interval}.parquet"
    else:
        raise ValueError(f"Unknown source: {source}")

    if not filepath.exists():
        print(f"Data file not found: {filepath}")
        print("Run 'python scripts/download_data.py' first to download data.")
        sys.exit(1)

    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} bars from {filepath}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    # Prepare DataFrame for BarDataWrangler
    # Required columns: open, high, low, close, volume (with timestamp index)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Select OHLCV columns
    bar_df = df[["open", "high", "low", "close", "volume"]].copy()
    bar_df = bar_df.dropna()

    # Ensure timestamp index is timezone-aware UTC
    if bar_df.index.tz is None:
        bar_df.index = bar_df.index.tz_localize("UTC")

    # Create bar type
    bar_type = BarType.from_str(
        f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"
    )

    # Use BarDataWrangler to create Bar objects
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    bars = wrangler.process(bar_df)

    print(f"Created {len(bars)} NautilusTrader Bar objects")
    if len(bars) > 0:
        print(f"First bar: {bars[0]}")
        print(f"Last bar:  {bars[-1]}")

    return bars, bar_type


def run_backtest(data_source: str = "trades", interval: str = "1min") -> None:
    """Run the Olympex strategy backtest."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OLYMPEX ADAPTIVE STRATEGY — BACKTEST ON REAL BTC DATA")
    print("=" * 70)

    # Create instrument
    instrument = create_btcusdt_instrument()
    print(f"\nInstrument: {instrument.id}")

    # Load data
    print(f"\nLoading data (source={data_source}, interval={interval})...")
    bars, bar_type = load_candle_data(instrument, source=data_source, interval=interval)

    if len(bars) < 100:
        print(f"WARNING: Only {len(bars)} bars. Need at least 100 for meaningful backtest.")
        if len(bars) < 30:
            print("Too few bars to run backtest. Exiting.")
            return

    # Configure engine
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="INFO"),
        ),
    )

    # Add venue
    HYPERLIQUID = Venue("HYPERLIQUID")
    engine.add_venue(
        venue=HYPERLIQUID,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
    )

    # Add instrument and data
    engine.add_instrument(instrument)
    engine.add_data(bars)

    # Create strategy
    strategy_config = OlympexStrategyConfig(
        bar_type=str(bar_type),
        instrument_id=str(instrument.id),
        ema_fast_period=10,
        ema_slow_period=21,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        macd_fast=12,
        macd_slow=26,
        atr_period=14,
        sl_atr_multiplier=2.0,
        rr_ratio=3.0,
        trailing_atr_multiplier=1.5,
        max_holding_bars=100,
        confidence_threshold=0.15,
        position_size=0.01,
        mr_rsi_buy_threshold=40.0,
        mr_rsi_sell_threshold=60.0,
    )
    strategy = OlympexStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Run backtest
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST...")
    print("=" * 70)
    engine.run()

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    # Account report
    print("\n--- Account Report ---")
    print(engine.trader.generate_account_report(HYPERLIQUID))

    # Order fills
    print("\n--- Order Fills ---")
    print(engine.trader.generate_order_fills_report())

    # Position report
    print("\n--- Positions ---")
    print(engine.trader.generate_positions_report())

    # Strategy stats
    print("\n--- Strategy Stats ---")
    print(f"Trend entries: {strategy.stats['trend_entries']}")
    print(f"Mean reversion entries: {strategy.stats['mr_entries']}")
    print(f"Total entries: {strategy.stats['trend_entries'] + strategy.stats['mr_entries']}")
    print(f"Exits - TP: {strategy.stats['exits_tp']}")
    print(f"Exits - SL: {strategy.stats['exits_sl']}")
    print(f"Exits - Trailing: {strategy.stats['exits_trailing']}")
    print(f"Exits - Timeout: {strategy.stats['exits_timeout']}")
    print(f"Exits - Regime Flip: {strategy.stats['exits_regime_flip']}")

    # Calculate additional metrics from positions
    positions_report = engine.trader.generate_positions_report()
    if len(positions_report) > 0:
        analyze_results(positions_report)

    # Save results
    results_file = RESULTS_DIR / f"backtest_{data_source}_{interval}.txt"
    with open(results_file, "w") as f:
        f.write(f"Olympex Backtest Results — {data_source} {interval}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Bars: {len(bars)}\n")
        f.write(f"Trend entries: {strategy.stats['trend_entries']}\n")
        f.write(f"MR entries: {strategy.stats['mr_entries']}\n")
        f.write(str(positions_report) + "\n")
    print(f"\nResults saved to: {results_file}")

    # Dispose engine
    engine.dispose()


def analyze_results(positions_df: pd.DataFrame) -> None:
    """Analyze position results and print summary metrics."""
    print("\n--- Performance Metrics ---")

    if "realized_pnl" not in positions_df.columns:
        print("No realized PnL data available")
        return

    # Parse PnL — may be string like "123.45 USDT"
    pnl_values = []
    for val in positions_df["realized_pnl"]:
        if isinstance(val, str):
            pnl_values.append(float(val.split()[0]))
        else:
            pnl_values.append(float(val))

    total_pnl = sum(pnl_values)
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]

    total_trades = len(pnl_values)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown (simple)
    cumulative = []
    running = 0
    peak = 0
    max_dd = 0
    for p in pnl_values:
        running += p
        cumulative.append(running)
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    print(f"Total trades: {total_trades}")
    print(f"Win/Loss: {win_count}/{loss_count}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f} USDT")
    print(f"Gross profit: {gross_profit:.2f} USDT")
    print(f"Gross loss: {gross_loss:.2f} USDT")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Max drawdown: {max_dd:.2f} USDT")
    if wins:
        print(f"Avg win: {gross_profit / win_count:.2f} USDT")
    if losses:
        print(f"Avg loss: {gross_loss / loss_count:.2f} USDT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Olympex Backtest Runner")
    parser.add_argument(
        "--data-source",
        choices=["candles", "trades"],
        default="trades",
        help="Data source: 'candles' (exchange OHLCV) or 'trades' (derived from ticks)",
    )
    parser.add_argument(
        "--interval",
        choices=["1min", "5min"],
        default="1min",
        help="Bar interval (only for trades source)",
    )
    args = parser.parse_args()
    run_backtest(data_source=args.data_source, interval=args.interval)
