"""
Olympex Master Strategy — Adaptive multi-regime trading system.

Architecture:
  MarketMeasure → RegimeClassifier → Strategy Modules → Order Management

The master strategy:
1. Feeds bar data to NautilusTrader indicators
2. Passes indicator values to MarketMeasure for state quantification
3. Passes MarketState to RegimeClassifier for regime detection
4. Routes to appropriate strategy modules based on regime
5. Manages orders via bracket orders (entry + SL + TP)
6. Handles position lifecycle (trailing stops, timeouts)
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import (
    AverageTrueRange,
    BollingerBands,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    MovingAverageType,
    RelativeStrengthIndex,
)
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

from .market_measure import MarketMeasure, MarketState
from .regime_classifier import Regime, RegimeClassifier, RegimeSignal
from .strategy_modules import (
    BreakoutModule,
    FadeModule,
    MeanReversionModule,
    ScalpModule,
    SignalType,
    TradeSignal,
    TrendModule,
)


class OlympexStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for Olympex master strategy."""

    bar_type: str
    instrument_id: str

    # Indicator periods
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    atr_period: int = 14

    # Position management
    position_size: float = 0.01
    max_holding_bars: int = 60
    trailing_atr_mult: float = 1.0

    # Signal gating
    min_signal_strength: float = 0.15  # minimum strength to take a trade

    # Gap detection
    gap_seconds: int = 300


class OlympexStrategy(Strategy):
    """
    Master strategy that measures the market, classifies regime,
    and routes to the appropriate strategy module.
    """

    def __init__(self, config: OlympexStrategyConfig) -> None:
        super().__init__(config)

        self.bar_type = BarType.from_str(config.bar_type)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.instrument: Instrument | None = None

        # NautilusTrader indicators (managed by engine)
        self.ema_fast = ExponentialMovingAverage(config.ema_fast_period)
        self.ema_slow = ExponentialMovingAverage(config.ema_slow_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.bb = BollingerBands(config.bb_period, config.bb_std)
        self.macd = MovingAverageConvergenceDivergence(
            config.macd_fast, config.macd_slow, MovingAverageType.EXPONENTIAL,
        )
        self.atr = AverageTrueRange(config.atr_period)

        # Core components
        self.market_measure = MarketMeasure()
        self.regime_classifier = RegimeClassifier()

        # Strategy modules
        self.trend_mod = TrendModule()
        self.mr_mod = MeanReversionModule()
        self.breakout_mod = BreakoutModule()
        self.scalp_mod = ScalpModule()
        self.fade_mod = FadeModule()

        # Position tracking
        self._prev_bar_ts: int = 0
        self._bar_count: int = 0
        self._entry_side: OrderSide | None = None
        self._entry_price: float = 0.0
        self._entry_signal_type: SignalType | None = None
        self._bars_since_entry: int = 0
        self._best_pnl: float = 0.0

        # Stats
        self.stats = {
            "trend_entries": 0,
            "mr_entries": 0,
            "breakout_entries": 0,
            "scalp_entries": 0,
            "fade_entries": 0,
            "exits_tp": 0,
            "exits_sl": 0,
            "exits_trailing": 0,
            "exits_timeout": 0,
            "gaps_detected": 0,
            "signals_filtered": 0,
            "regime_trending": 0,
            "regime_ranging": 0,
            "regime_breakout": 0,
            "regime_squeeze": 0,
            "regime_volatile": 0,
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found")
            return

        for ind in [self.ema_fast, self.ema_slow, self.rsi, self.bb, self.macd, self.atr]:
            self.register_indicator_for_bars(self.bar_type, ind)

        self.subscribe_bars(self.bar_type)
        self.log.info("OlympexStrategy (multi-regime) started")

    def on_bar(self, bar: Bar) -> None:
        self._bar_count += 1

        # Gap detection
        if self._prev_bar_ts > 0:
            gap_s = (bar.ts_event - self._prev_bar_ts) / 1_000_000_000
            if gap_s > self.config.gap_seconds:
                self._on_gap(gap_s)
        self._prev_bar_ts = bar.ts_event

        # Wait for indicators
        if not self._indicators_ready():
            return

        # Step 1: Measure market
        state = self.market_measure.update(
            open_=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            atr_value=self.atr.value,
            rsi_value=self.rsi.value,
            macd_value=self.macd.value,
            macd_signal=self.macd.average.value if hasattr(self.macd, 'average') else self.macd.value,
            ema_fast=self.ema_fast.value,
            ema_slow=self.ema_slow.value,
            bb_upper=self.bb.upper,
            bb_lower=self.bb.lower,
            bb_middle=self.bb.middle,
        )

        if not state.is_valid:
            return

        # Step 2: Classify regime
        signal = self.regime_classifier.classify(state)
        self._track_regime(signal)

        # Step 3: Route to strategies or manage position
        price = float(bar.close)
        atr_val = self.atr.value
        is_flat = self.portfolio.is_flat(self.instrument_id)

        if not is_flat:
            self._bars_since_entry += 1
            self._manage_position(price, atr_val, signal)
        else:
            self._route_signals(bar, state, signal, price, atr_val)

    # -------------------------------------------------------------------------
    # Signal routing
    # -------------------------------------------------------------------------
    def _route_signals(
        self, bar: Bar, state: MarketState, signal: RegimeSignal,
        price: float, atr: float,
    ) -> None:
        """Collect signals from all active modules and pick the best one."""
        candidates: list[TradeSignal] = []

        # Check each module — only if its activation score is meaningful
        if signal.trend_score > 0.2:
            s = self.trend_mod.check_entry(
                state, signal, self.ema_fast.value, self.ema_slow.value,
            )
            if s:
                candidates.append(s)

        if signal.mr_score > 0.2:
            s = self.mr_mod.check_entry(
                state, signal, price, self.bb.upper, self.bb.lower,
            )
            if s:
                candidates.append(s)

        if signal.breakout_score > 0.2:
            s = self.breakout_mod.check_entry(
                state, signal, price, float(bar.high), float(bar.low),
            )
            if s:
                candidates.append(s)

        if signal.scalp_score > 0.3:
            s = self.scalp_mod.check_entry(
                state, signal, price, float(bar.open),
            )
            if s:
                candidates.append(s)

        if signal.fade_score > 0.3:
            s = self.fade_mod.check_entry(state, signal)
            if s:
                candidates.append(s)

        # Also update trend module even if no signal (for EMA tracking)
        if signal.trend_score <= 0.2:
            self.trend_mod._update(self.ema_fast.value, self.ema_slow.value)
        # Always update breakout history
        if signal.breakout_score <= 0.2:
            self.breakout_mod._high_history.append(float(bar.high))
            self.breakout_mod._low_history.append(float(bar.low))
            if len(self.breakout_mod._high_history) > self.breakout_mod.lookback + 1:
                self.breakout_mod._high_history = self.breakout_mod._high_history[-(self.breakout_mod.lookback + 1):]
                self.breakout_mod._low_history = self.breakout_mod._low_history[-(self.breakout_mod.lookback + 1):]

        if not candidates:
            return

        # Pick highest strength signal
        best = max(candidates, key=lambda s: s.strength)

        if best.strength < self.config.min_signal_strength:
            self.stats["signals_filtered"] += 1
            return

        # Execute
        self._enter_bracket(best, price, atr, signal)

    # -------------------------------------------------------------------------
    # Order execution
    # -------------------------------------------------------------------------
    def _enter_bracket(
        self, sig: TradeSignal, price: float, atr: float, regime: RegimeSignal,
    ) -> None:
        sl_distance = atr * sig.sl_atr_mult
        tp_distance = atr * sig.tp_atr_mult

        if sig.side == OrderSide.BUY:
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance

        sl_price_obj = self.instrument.make_price(sl_price)
        tp_price_obj = self.instrument.make_price(tp_price)
        qty = self.instrument.make_qty(self.config.position_size)

        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=sig.side,
            quantity=qty,
            sl_trigger_price=sl_price_obj,
            tp_price=tp_price_obj,
        )
        self.submit_order_list(bracket)

        self._entry_side = sig.side
        self._entry_price = price
        self._entry_signal_type = sig.signal_type
        self._bars_since_entry = 0
        self._best_pnl = 0.0

        # Track stats
        stat_key = f"{sig.signal_type.value}_entries"
        if stat_key in self.stats:
            self.stats[stat_key] += 1

        self.log.info(
            f"ENTRY: {sig.reason} | {sig.side.name} @ {price:.1f} | "
            f"SL={float(sl_price_obj):.1f} TP={float(tp_price_obj):.1f} | "
            f"regime={regime.regime.value} conf={regime.confidence:.2f} str={sig.strength:.2f}"
        )

    # -------------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------------
    def _manage_position(self, price: float, atr: float, signal: RegimeSignal) -> None:
        if self._entry_side is None:
            return

        if self._entry_side == OrderSide.BUY:
            pnl = price - self._entry_price
        else:
            pnl = self._entry_price - price

        if pnl > self._best_pnl:
            self._best_pnl = pnl

        # Trailing stop
        trail_dist = atr * self.config.trailing_atr_mult
        if self._best_pnl > trail_dist * 2 and (self._best_pnl - pnl) > trail_dist:
            self._close_and_cancel("TRAILING", price, pnl)
            self.stats["exits_trailing"] += 1
            return

        # Timeout
        if self._bars_since_entry >= self.config.max_holding_bars:
            self._close_and_cancel("TIMEOUT", price, pnl)
            self.stats["exits_timeout"] += 1
            return

    def _close_and_cancel(self, reason: str, price: float, pnl: float) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info(
            f"EXIT: {reason} @ {price:.1f} | PnL={pnl:.1f} | "
            f"type={self._entry_signal_type.value if self._entry_signal_type else '?'} | "
            f"bars={self._bars_since_entry}"
        )
        self._reset_entry()

    def _reset_entry(self) -> None:
        self._entry_side = None
        self._entry_price = 0.0
        self._entry_signal_type = None
        self._bars_since_entry = 0
        self._best_pnl = 0.0

    # -------------------------------------------------------------------------
    # Order events
    # -------------------------------------------------------------------------
    def on_order_filled(self, event) -> None:
        if self.portfolio.is_flat(self.instrument_id):
            order = self.cache.order(event.client_order_id)
            if order and self._entry_side is not None:
                if order.order_type.name == "STOP_MARKET":
                    self.stats["exits_sl"] += 1
                    self.log.info(f"EXIT: SL (bracket) @ {float(event.last_px):.1f}")
                elif order.order_type.name == "LIMIT" and order.is_reduce_only:
                    self.stats["exits_tp"] += 1
                    self.log.info(f"EXIT: TP (bracket) @ {float(event.last_px):.1f}")
                self._reset_entry()

    # -------------------------------------------------------------------------
    # Gap handling
    # -------------------------------------------------------------------------
    def _on_gap(self, gap_seconds: float) -> None:
        self.stats["gaps_detected"] += 1
        self.log.info(f"Gap detected: {gap_seconds:.0f}s — resetting state")

        # Reset all components
        self.market_measure.reset()
        self.regime_classifier.reset()
        self.trend_mod.reset()
        self.mr_mod.reset()
        self.breakout_mod.reset()
        self.scalp_mod.reset()
        self.fade_mod.reset()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _indicators_ready(self) -> bool:
        return all(ind.initialized for ind in [
            self.ema_fast, self.ema_slow, self.rsi, self.bb, self.macd, self.atr,
        ])

    def _track_regime(self, signal: RegimeSignal) -> None:
        key = f"regime_{signal.regime.value}"
        if key in self.stats:
            self.stats[key] += 1

    def on_stop(self) -> None:
        self.log.info("=" * 60)
        self.log.info("OLYMPEX MULTI-REGIME STRATEGY — FINAL STATISTICS")
        self.log.info("=" * 60)
        self.log.info(f"Total bars: {self._bar_count}")
        self.log.info(f"Gaps: {self.stats['gaps_detected']}")
        self.log.info(f"--- Regime Distribution ---")
        total_regime = sum(self.stats[f"regime_{r.value}"] for r in Regime)
        for r in Regime:
            count = self.stats[f"regime_{r.value}"]
            pct = 100 * count / total_regime if total_regime > 0 else 0
            self.log.info(f"  {r.value:12s}: {count:5d} bars ({pct:.1f}%)")
        self.log.info(f"--- Entries by Type ---")
        for t in ["trend", "mr", "breakout", "scalp", "fade"]:
            self.log.info(f"  {t:12s}: {self.stats[f'{t}_entries']}")
        total = sum(self.stats[f"{t}_entries"] for t in ["trend", "mr", "breakout", "scalp", "fade"])
        self.log.info(f"  {'total':12s}: {total}")
        self.log.info(f"  filtered:     {self.stats['signals_filtered']}")
        self.log.info(f"--- Exits ---")
        self.log.info(f"  TP:       {self.stats['exits_tp']}")
        self.log.info(f"  SL:       {self.stats['exits_sl']}")
        self.log.info(f"  Trailing: {self.stats['exits_trailing']}")
        self.log.info(f"  Timeout:  {self.stats['exits_timeout']}")
        self.log.info("=" * 60)
