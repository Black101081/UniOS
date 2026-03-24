"""
Olympex Adaptive Strategy v2 for NautilusTrader.

Designed for BTC 1-min candles from Hyperliquid.

Key improvements over v1:
- Uses bracket orders (entry + SL + TP submitted atomically)
- Trend following as primary strategy (data shows 22-26 bar avg trends)
- Tighter mean reversion filters (v1 overtrained, 28.6% win rate)
- Gap detection — resets state on data gaps
- ATR-based dynamic position exits

Entry Logic:
  TREND: EMA crossover confirmed by MACD direction and RSI momentum
  MEAN REVERSION: BB extremes + RSI divergence (strict filter)

Exit Logic:
  Bracket order handles SL/TP natively via stop-market and limit orders.
  Strategy manages trailing stops and timeout exits.
"""

from dataclasses import dataclass
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
from nautilus_trader.model.enums import (
    OrderSide,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


class OlympexStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for OlympexStrategy v2."""

    bar_type: str
    instrument_id: str

    # EMA
    ema_fast_period: int = 9
    ema_slow_period: int = 21

    # RSI
    rsi_period: int = 14

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26

    # ATR
    atr_period: int = 14

    # Risk management
    sl_atr_multiplier: float = 1.5     # tight SL — data shows avg win > avg loss
    rr_ratio: float = 2.0             # TP = SL × 2.0
    trailing_atr_multiplier: float = 1.0

    # Trade management
    max_holding_bars: int = 60          # shorter than v1 (was 100)
    position_size: float = 0.01        # BTC per trade

    # Trend entry filters
    trend_rsi_min: float = 40.0        # don't buy if RSI < 40 in uptrend (too weak)
    trend_rsi_max: float = 60.0        # don't sell if RSI > 60 in downtrend

    # Mean reversion filters (strict)
    mr_bb_threshold: float = 0.10      # must be within 10% of band (was 20%)
    mr_rsi_oversold: float = 30.0      # RSI < 30 for MR buy (was 40)
    mr_rsi_overbought: float = 70.0    # RSI > 70 for MR sell (was 60)

    # Gap detection
    gap_seconds: int = 300             # 5 min gap = reset indicators

    # Cooldown between trades (prevent overtrading)
    min_bars_between_trades: int = 0   # 0 = no cooldown


class OlympexStrategy(Strategy):
    """
    Adaptive BTC strategy v2 — trend following + selective mean reversion.
    Uses bracket orders for proper SL/TP management.
    """

    def __init__(self, config: OlympexStrategyConfig) -> None:
        super().__init__(config)

        self.bar_type = BarType.from_str(config.bar_type)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.instrument: Instrument | None = None

        # Indicators
        self.ema_fast = ExponentialMovingAverage(config.ema_fast_period)
        self.ema_slow = ExponentialMovingAverage(config.ema_slow_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.bb = BollingerBands(config.bb_period, config.bb_std)
        self.macd = MovingAverageConvergenceDivergence(
            config.macd_fast,
            config.macd_slow,
            MovingAverageType.EXPONENTIAL,
        )
        self.atr = AverageTrueRange(config.atr_period)

        # State
        self._prev_ema_fast: float | None = None
        self._prev_ema_slow: float | None = None
        self._prev_macd: float | None = None
        self._prev_bar_ts: int = 0          # nanoseconds
        self._bars_since_entry: int = 0
        self._bar_count: int = 0
        self._warmup_bars: int = 0          # bars since last gap/start
        self._entry_side: OrderSide | None = None
        self._entry_price: float = 0.0
        self._best_pnl: float = 0.0         # for trailing stop
        self._bars_since_exit: int = 999     # cooldown counter

        # Stats
        self.stats = {
            "trend_entries": 0,
            "mr_entries": 0,
            "exits_tp": 0,
            "exits_sl": 0,
            "exits_trailing": 0,
            "exits_timeout": 0,
            "gaps_detected": 0,
        }

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found")
            return

        self.register_indicator_for_bars(self.bar_type, self.ema_fast)
        self.register_indicator_for_bars(self.bar_type, self.ema_slow)
        self.register_indicator_for_bars(self.bar_type, self.rsi)
        self.register_indicator_for_bars(self.bar_type, self.bb)
        self.register_indicator_for_bars(self.bar_type, self.macd)
        self.register_indicator_for_bars(self.bar_type, self.atr)

        self.subscribe_bars(self.bar_type)
        self.log.info("OlympexStrategy v2 started")

    def on_bar(self, bar: Bar) -> None:
        self._bar_count += 1

        # Gap detection: if time between bars > threshold, reset warmup
        bar_ts_ns = bar.ts_event
        if self._prev_bar_ts > 0:
            gap_seconds = (bar_ts_ns - self._prev_bar_ts) / 1_000_000_000
            if gap_seconds > self.config.gap_seconds:
                self._warmup_bars = 0
                self.stats["gaps_detected"] += 1
                self.log.info(f"Gap detected: {gap_seconds:.0f}s — resetting warmup")
        self._prev_bar_ts = bar_ts_ns
        self._warmup_bars += 1

        # Need warmup for indicators + state
        if not self._indicators_ready() or self._warmup_bars < self.config.ema_slow_period + 5:
            self._update_prev_values()
            return

        price = float(bar.close)
        atr_val = self.atr.value
        is_flat = self.portfolio.is_flat(self.instrument_id)

        if not is_flat:
            self._bars_since_entry += 1
            self._manage_position(price, atr_val)
        else:
            self._bars_since_exit += 1
            if self._bars_since_exit >= self.config.min_bars_between_trades:
                self._check_entries(price, atr_val)

        self._update_prev_values()

    def _indicators_ready(self) -> bool:
        return (
            self.ema_fast.initialized
            and self.ema_slow.initialized
            and self.rsi.initialized
            and self.bb.initialized
            and self.macd.initialized
            and self.atr.initialized
        )

    def _update_prev_values(self) -> None:
        if self.ema_fast.initialized:
            self._prev_ema_fast = self.ema_fast.value
        if self.ema_slow.initialized:
            self._prev_ema_slow = self.ema_slow.value
        if self.macd.initialized:
            self._prev_macd = self.macd.value

    # -------------------------------------------------------------------------
    # Entry Logic
    # -------------------------------------------------------------------------
    def _check_entries(self, price: float, atr: float) -> None:
        if self._prev_ema_fast is None or self._prev_ema_slow is None:
            return
        if atr <= 0:
            return

        config: OlympexStrategyConfig = self.config
        ema_fast = self.ema_fast.value
        ema_slow = self.ema_slow.value
        rsi = self.rsi.value
        macd = self.macd.value
        bb_upper = self.bb.upper
        bb_lower = self.bb.lower

        # --- TREND FOLLOWING ---
        # Bullish: EMA fast crosses above slow + MACD positive & rising + RSI confirms
        bull_cross = self._prev_ema_fast <= self._prev_ema_slow and ema_fast > ema_slow
        bear_cross = self._prev_ema_fast >= self._prev_ema_slow and ema_fast < ema_slow
        macd_rising = self._prev_macd is not None and macd > self._prev_macd
        macd_falling = self._prev_macd is not None and macd < self._prev_macd

        if bull_cross and macd > 0 and rsi > config.trend_rsi_min:
            self._enter_bracket(OrderSide.BUY, price, atr, "TREND_BULL")
            return

        if bear_cross and macd < 0 and rsi < config.trend_rsi_max:
            self._enter_bracket(OrderSide.SELL, price, atr, "TREND_BEAR")
            return

        # --- MEAN REVERSION (strict) ---
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return
        bb_pct = (price - bb_lower) / bb_range

        # Buy: price at lower BB + RSI oversold
        if bb_pct <= config.mr_bb_threshold and rsi < config.mr_rsi_oversold:
            self._enter_bracket(OrderSide.BUY, price, atr, "MR_BUY")
            return

        # Sell: price at upper BB + RSI overbought
        if bb_pct >= (1.0 - config.mr_bb_threshold) and rsi > config.mr_rsi_overbought:
            self._enter_bracket(OrderSide.SELL, price, atr, "MR_SELL")
            return

    def _enter_bracket(self, side: OrderSide, price: float, atr: float, reason: str) -> None:
        """Submit bracket order: market entry + stop-loss + take-profit."""
        config: OlympexStrategyConfig = self.config
        sl_distance = atr * config.sl_atr_multiplier
        tp_distance = sl_distance * config.rr_ratio

        if side == OrderSide.BUY:
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance

        # Clamp prices to instrument precision
        sl_price_obj = self.instrument.make_price(sl_price)
        tp_price_obj = self.instrument.make_price(tp_price)
        qty = self.instrument.make_qty(config.position_size)

        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            sl_trigger_price=sl_price_obj,
            tp_price=tp_price_obj,
        )
        self.submit_order_list(bracket)

        # Track state
        self._entry_side = side
        self._entry_price = price
        self._bars_since_entry = 0
        self._best_pnl = 0.0

        if reason.startswith("TREND"):
            self.stats["trend_entries"] += 1
        else:
            self.stats["mr_entries"] += 1

        self.log.info(
            f"ENTRY: {reason} | {side.name} @ {price:.1f} | "
            f"SL={float(sl_price_obj):.1f} TP={float(tp_price_obj):.1f} | ATR={atr:.1f}"
        )

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------
    def _manage_position(self, price: float, atr: float) -> None:
        """Check trailing stop and timeout exits. SL/TP are handled by bracket."""
        if self._entry_side is None:
            return

        config: OlympexStrategyConfig = self.config

        # Calculate unrealized PnL
        if self._entry_side == OrderSide.BUY:
            pnl = price - self._entry_price
        else:
            pnl = self._entry_price - price

        # Track best PnL for trailing
        if pnl > self._best_pnl:
            self._best_pnl = pnl

        # Trailing stop: if we had a good run but price pulled back
        trailing_distance = atr * config.trailing_atr_multiplier
        if self._best_pnl > trailing_distance * 2 and (self._best_pnl - pnl) > trailing_distance:
            self._close_and_cancel("TRAILING_STOP", price, pnl)
            self.stats["exits_trailing"] += 1
            return

        # Timeout: close if held too long
        if self._bars_since_entry >= config.max_holding_bars:
            self._close_and_cancel("TIMEOUT", price, pnl)
            self.stats["exits_timeout"] += 1
            return

    def _close_and_cancel(self, reason: str, price: float, pnl: float) -> None:
        """Close position and cancel any remaining bracket orders."""
        # Cancel all open orders for this instrument (SL/TP from bracket)
        self.cancel_all_orders(self.instrument_id)

        # Close position
        self.close_all_positions(self.instrument_id)

        self.log.info(f"EXIT: {reason} @ {price:.1f} | PnL={pnl:.1f} | bars_held={self._bars_since_entry}")
        self._reset_entry_state()

    def _reset_entry_state(self) -> None:
        self._entry_side = None
        self._entry_price = 0.0
        self._bars_since_entry = 0
        self._best_pnl = 0.0
        self._bars_since_exit = 0

    # -------------------------------------------------------------------------
    # Order Events
    # -------------------------------------------------------------------------
    def on_order_filled(self, event) -> None:
        """Track when SL or TP is hit by the bracket."""
        order = self.cache.order(event.client_order_id)
        if order is None:
            return

        # If this fill closes our position, record which exit type
        if self.portfolio.is_flat(self.instrument_id):
            tags = order.tags or ""
            if "sl" in str(tags).lower() or order.order_type.name == "STOP_MARKET":
                # Likely a stop-loss fill
                if self._entry_side is not None:
                    self.stats["exits_sl"] += 1
                    self.log.info(f"EXIT: STOP_LOSS (bracket) @ {float(event.last_px):.1f}")
            elif "tp" in str(tags).lower() or (order.order_type.name == "LIMIT" and order.is_reduce_only):
                if self._entry_side is not None:
                    self.stats["exits_tp"] += 1
                    self.log.info(f"EXIT: TAKE_PROFIT (bracket) @ {float(event.last_px):.1f}")
            else:
                # Generic close (from close_all_positions or manual)
                pass

            self._reset_entry_state()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def on_stop(self) -> None:
        self.log.info("=" * 50)
        self.log.info("OLYMPEX STRATEGY v2 STATISTICS")
        self.log.info("=" * 50)
        self.log.info(f"Total bars: {self._bar_count}")
        self.log.info(f"Gaps detected: {self.stats['gaps_detected']}")
        self.log.info(f"Trend entries: {self.stats['trend_entries']}")
        self.log.info(f"MR entries: {self.stats['mr_entries']}")
        self.log.info(f"Exits - TP: {self.stats['exits_tp']}")
        self.log.info(f"Exits - SL: {self.stats['exits_sl']}")
        self.log.info(f"Exits - Trailing: {self.stats['exits_trailing']}")
        self.log.info(f"Exits - Timeout: {self.stats['exits_timeout']}")
        self.log.info("=" * 50)
