"""
Sniper 33-dim Backtest Runner — NautilusTrader backtest on real BTC data.

Usage:
    python run_sniper_backtest.py [--data binance|hyperliquid] [--tp 50] [--sl 150] [--condition storm] [--direction AUTO]

Data sources:
    - binance: 1.7M+ 1-min candles (2023-2026) — full test
    - hyperliquid: ~12K 1-min candles (30 days) — quick test
"""

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler

from strategies.sniper_strategy import SniperStrategy, SniperStrategyConfig

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def create_instrument(venue_name: str = "HYPERLIQUID") -> CryptoPerpetual:
    """Create a BTC-USD perpetual instrument."""
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str(f"BTC-USD-PERP.{venue_name}"),
        raw_symbol=Symbol("BTC-USD-PERP"),
        base_currency=Currency.from_str("BTC"),
        quote_currency=Currency.from_str("USD"),
        settlement_currency=Currency.from_str("USDC"),
        is_inverse=False,
        price_precision=1,
        size_precision=5,
        price_increment=Price.from_str("0.1"),
        size_increment=Quantity.from_str("0.00001"),
        max_quantity=Quantity.from_str("1000.00000"),
        min_quantity=Quantity.from_str("0.00001"),
        max_notional=None,
        min_notional=Money(10.00, Currency.from_str("USDC")),
        max_price=Price.from_str("1000000.0"),
        min_price=Price.from_str("0.1"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
        ts_event=0,
        ts_init=0,
    )


def load_data(source: str, instrument: CryptoPerpetual, limit: int = 0) -> tuple[list[Bar], BarType, pd.DataFrame]:
    """Load bar data from parquet files."""
    if source == "binance":
        filepath = DATA_DIR / "btc_binance_1min.parquet"
    else:
        filepath = DATA_DIR / "btc_candles_1min.parquet"

    if not filepath.exists():
        print(f"ERROR: Data file not found: {filepath}")
        sys.exit(1)

    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} bars from {filepath.name}")

    df = df.sort_values("timestamp").reset_index(drop=True)

    if limit > 0 and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
        print(f"Using last {limit:,} bars")

    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():,.0f} to ${df['close'].max():,.0f}")

    # Prepare for NautilusTrader
    bar_df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
    bar_df = bar_df.dropna()

    if bar_df.index.tz is None:
        bar_df.index = bar_df.index.tz_localize("UTC")

    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    bars = wrangler.process(bar_df)

    print(f"Created {len(bars):,} NautilusTrader Bar objects")
    return bars, bar_type, df


def analyze_positions(positions_df: pd.DataFrame) -> dict:
    """Analyze position results and return metrics."""
    metrics = {}

    if positions_df is None or len(positions_df) == 0:
        return metrics

    if "realized_pnl" not in positions_df.columns:
        return metrics

    pnl_values = []
    for val in positions_df["realized_pnl"]:
        if isinstance(val, str):
            pnl_values.append(float(val.split()[0]))
        else:
            pnl_values.append(float(val))

    pnls = np.array(pnl_values)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    metrics["total_trades"] = len(pnls)
    metrics["wins"] = len(wins)
    metrics["losses"] = len(losses)
    metrics["win_rate"] = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
    metrics["total_pnl"] = float(pnls.sum())
    metrics["avg_pnl"] = float(pnls.mean()) if len(pnls) > 0 else 0
    metrics["gross_profit"] = float(wins.sum()) if len(wins) > 0 else 0
    metrics["gross_loss"] = float(abs(losses.sum())) if len(losses) > 0 else 0
    metrics["profit_factor"] = (
        metrics["gross_profit"] / metrics["gross_loss"]
        if metrics["gross_loss"] > 0
        else float("inf")
    )
    metrics["avg_win"] = float(wins.mean()) if len(wins) > 0 else 0
    metrics["avg_loss"] = float(abs(losses.mean())) if len(losses) > 0 else 0

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    metrics["max_drawdown"] = float(dd.min())

    # Sharpe (annualized, assuming 1-min bars)
    if len(pnls) > 1 and np.std(pnls) > 0:
        # Approximate: trades per year / sqrt(trades per year)
        metrics["sharpe"] = float(np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)))
    else:
        metrics["sharpe"] = 0.0

    return metrics


def run_backtest(
    source: str = "hyperliquid",
    tp: float = 50.0,
    sl: float = 150.0,
    condition: str = "storm",
    direction: str = "AUTO",
    position_size: float = 0.01,
    max_hold: int = 120,
    cooldown: int = 5,
    calibration: int = 2000,
    limit: int = 0,
) -> None:
    """Run the Sniper 33-dim strategy backtest."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    venue_name = "BINANCE" if source == "binance" else "HYPERLIQUID"

    print("=" * 70)
    print("SNIPER 33-DIM STRATEGY — NAUTILUS BACKTEST")
    print("=" * 70)
    print(f"Config: TP=${tp} SL=${sl} | condition={condition} | direction={direction}")
    print(f"Position: {position_size} BTC | max_hold={max_hold} bars | cooldown={cooldown}")
    print(f"Data source: {source}")
    print()

    instrument = create_instrument(venue_name)
    bars, bar_type, raw_df = load_data(source, instrument, limit=limit)

    if len(bars) < 500:
        print(f"ERROR: Only {len(bars)} bars — need at least 500 for sniper strategy.")
        return

    # Configure engine
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="INFO"),
        ),
    )

    venue = Venue(venue_name)
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(10_000, Currency.from_str("USDC"))],
    )

    engine.add_instrument(instrument)
    engine.add_data(bars)

    # Create strategy
    strategy_config = SniperStrategyConfig(
        bar_type=str(bar_type),
        instrument_id=str(instrument.id),
        position_size=position_size,
        tp_price=tp,
        sl_price=sl,
        max_hold_bars=max_hold,
        cooldown_bars=cooldown,
        condition=condition,
        direction=direction,
        calibration_bars=calibration,
    )
    strategy = SniperStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Run
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST...")
    print("=" * 70)
    engine.run()

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    # Account
    print("\n--- Account Report ---")
    print(engine.trader.generate_account_report(venue))

    # Orders
    print("\n--- Order Fills ---")
    fills_report = engine.trader.generate_order_fills_report()
    print(f"Total fills: {len(fills_report)}")
    if len(fills_report) > 0:
        print(fills_report.tail(10))

    # Positions
    print("\n--- Positions ---")
    positions_report = engine.trader.generate_positions_report()
    print(f"Total positions: {len(positions_report)}")

    # Performance metrics
    if len(positions_report) > 0:
        metrics = analyze_positions(positions_report)
        print("\n--- Performance Metrics ---")
        print(f"Total trades:   {metrics.get('total_trades', 0)}")
        print(f"Win/Loss:       {metrics.get('wins', 0)}/{metrics.get('losses', 0)}")
        print(f"Win rate:       {metrics.get('win_rate', 0):.1f}%")
        print(f"Total PnL:      ${metrics.get('total_pnl', 0):+,.2f} USDC")
        print(f"Avg PnL/trade:  ${metrics.get('avg_pnl', 0):+,.2f}")
        print(f"Gross profit:   ${metrics.get('gross_profit', 0):,.2f}")
        print(f"Gross loss:     ${metrics.get('gross_loss', 0):,.2f}")
        print(f"Profit factor:  {metrics.get('profit_factor', 0):.2f}")
        print(f"Avg win:        ${metrics.get('avg_win', 0):,.2f}")
        print(f"Avg loss:       ${metrics.get('avg_loss', 0):,.2f}")
        print(f"Max drawdown:   ${metrics.get('max_drawdown', 0):,.2f}")
        print(f"Sharpe ratio:   {metrics.get('sharpe', 0):.2f}")
    else:
        metrics = {}

    # Strategy internal stats
    print("\n--- Strategy Internal Stats ---")
    s = strategy.stats
    for key, val in s.items():
        print(f"  {key}: {val}")

    # Save results
    results_file = RESULTS_DIR / f"sniper_{source}_tp{int(tp)}_sl{int(sl)}_{condition}_{direction}.txt"
    with open(results_file, "w") as f:
        f.write(f"Sniper 33-dim Backtest — {source} data\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config: TP=${tp} SL=${sl} condition={condition} direction={direction}\n")
        f.write(f"Bars: {len(bars):,}\n")
        f.write(f"Position size: {position_size} BTC\n\n")

        f.write("Strategy Stats:\n")
        for key, val in s.items():
            f.write(f"  {key}: {val}\n")

        if metrics:
            f.write("\nPerformance Metrics:\n")
            for key, val in metrics.items():
                f.write(f"  {key}: {val}\n")

        f.write(f"\nPositions Report:\n{positions_report}\n")

    print(f"\nResults saved to: {results_file}")

    engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper 33-dim Backtest Runner")
    parser.add_argument("--data", choices=["binance", "hyperliquid"], default="hyperliquid",
                        help="Data source (default: hyperliquid)")
    parser.add_argument("--tp", type=float, default=50.0, help="Take profit in USD (default: 50)")
    parser.add_argument("--sl", type=float, default=150.0, help="Stop loss in USD (default: 150)")
    parser.add_argument("--condition", default="storm",
                        choices=["ultra", "storm", "decouple", "quiet", "relaxed"],
                        help="Entry condition (default: storm)")
    parser.add_argument("--direction", default="AUTO",
                        choices=["LONG", "SHORT", "AUTO"],
                        help="Trade direction (default: AUTO)")
    parser.add_argument("--size", type=float, default=0.01, help="Position size in BTC (default: 0.01)")
    parser.add_argument("--max-hold", type=int, default=120, help="Max hold bars (default: 120)")
    parser.add_argument("--cooldown", type=int, default=5, help="Cooldown bars (default: 5)")
    parser.add_argument("--calibration", type=int, default=2000, help="Calibration bars (default: 2000)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to last N bars (default: 0 = all)")
    args = parser.parse_args()

    run_backtest(
        source=args.data,
        tp=args.tp,
        sl=args.sl,
        condition=args.condition,
        direction=args.direction,
        position_size=args.size,
        max_hold=args.max_hold,
        cooldown=args.cooldown,
        calibration=args.calibration,
        limit=args.limit,
    )
