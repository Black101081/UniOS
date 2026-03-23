"""
Olympex Adaptive Strategy for NautilusTrader.

Combines trend-following and mean-reversion approaches,
automatically switching based on market regime detection.

Entry Logic:
  - Trending market: EMA crossover + regime confirmation
  - Ranging market: Bollinger Band bounce + RSI confirmation

Exit Logic:
  - Take Profit: ATR × SL_multiplier × R:R ratio
  - Stop Loss: ATR × SL_multiplier
  - Trailing Stop: ATR × trailing_multiplier
  - Timeout: max holding bars exceeded
  - Regime Flip: exit trend trades when regime reverses
"""

from dataclasses import dataclass, field
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

from .regime_detector import MarketRegimeDetector, RegimeState


class OlympexStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for OlympexStrategy."""

    bar_type: str
    instrument_id: str

    # EMA parameters
    ema_fast_period: int = 10
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
    sl_atr_multiplier: float = 2.0
    rr_ratio: float = 3.0
    trailing_atr_multiplier: float = 1.5

    # Trade management
    max_holding_bars: int = 100
    confidence_threshold: float = 0.15
    position_size: float = 0.01  # BTC quantity per trade

    # Mean reversion thresholds (relaxed to fix no-trigger issue)
    mr_rsi_buy_threshold: float = 40.0
    mr_rsi_sell_threshold: float = 60.0


class OlympexStrategy(Strategy):
    """
    Adaptive strategy that switches between trend-following
    and mean-reversion based on market regime.
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

        # Regime detector
        self.regime_detector = MarketRegimeDetector()

        # State tracking
        self._prev_ema_fast: float | None = None
        self._prev_ema_slow: float | None = None
        self._entry_price: float | None = None
        self._entry_side: OrderSide | None = None
        self._entry_bar_count: int = 0
        self._bar_count: int = 0
        self._current_regime: RegimeState | None = None
        self._entry_regime_trending: bool = False

        # Stats
        self.stats = {
            "trend_entries": 0,
            "mr_entries": 0,
            "exits_tp": 0,
            "exits_sl": 0,
            "exits_trailing": 0,
            "exits_timeout": 0,
            "exits_regime_flip": 0,
        }

    def on_start(self) -> None:
        """Register indicators and subscribe to bar data."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found")
            return

        # Register indicators for automatic updates
        self.register_indicator_for_bars(self.bar_type, self.ema_fast)
        self.register_indicator_for_bars(self.bar_type, self.ema_slow)
        self.register_indicator_for_bars(self.bar_type, self.rsi)
        self.register_indicator_for_bars(self.bar_type, self.bb)
        self.register_indicator_for_bars(self.bar_type, self.macd)
        self.register_indicator_for_bars(self.bar_type, self.atr)

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)
        self.log.info("OlympexStrategy started")

    def on_bar(self, bar: Bar) -> None:
        """Main strategy logic — called on each new bar."""
        self._bar_count += 1

        # Wait for all indicators to be initialized
        if not self._indicators_ready():
            return

        # Get current indicator values
        price = float(bar.close)
        rsi_val = self.rsi.value
        macd_val = self.macd.value
        ema_fast_val = self.ema_fast.value
        ema_slow_val = self.ema_slow.value
        atr_val = self.atr.value
        bb_upper = self.bb.upper
        bb_lower = self.bb.lower
        bb_middle = self.bb.middle

        # Compute regime
        regime = self.regime_detector.compute(
            rsi_value=rsi_val,
            macd_value=macd_val,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_middle=bb_middle,
            price=price,
            ema_fast=ema_fast_val,
            ema_slow=ema_slow_val,
            atr_value=atr_val,
        )
        self._current_regime = regime

        # Check position state
        is_flat = self.portfolio.is_flat(self.instrument_id)

        if not is_flat:
            # Check exits
            self._check_exits(price, atr_val, regime)
        else:
            # Check entries
            self._check_entries(price, rsi_val, ema_fast_val, ema_slow_val, bb_upper, bb_lower, atr_val, regime)

        # Update previous EMA values for crossover detection
        self._prev_ema_fast = ema_fast_val
        self._prev_ema_slow = ema_slow_val

    def _indicators_ready(self) -> bool:
        """Check if all indicators are initialized."""
        return (
            self.ema_fast.initialized
            and self.ema_slow.initialized
            and self.rsi.initialized
            and self.bb.initialized
            and self.macd.initialized
            and self.atr.initialized
        )

    def _check_entries(
        self,
        price: float,
        rsi: float,
        ema_fast: float,
        ema_slow: float,
        bb_upper: float,
        bb_lower: float,
        atr: float,
        regime: RegimeState,
    ) -> None:
        """Check for trade entry signals."""

        # Need previous EMA values for crossover detection
        if self._prev_ema_fast is None or self._prev_ema_slow is None:
            return

        config: OlympexStrategyConfig = self.config

        # TREND FOLLOWING: EMA crossover + regime confirmation
        if regime.is_trending:
            # Bullish crossover: fast crosses above slow
            bull_cross = (
                self._prev_ema_fast <= self._prev_ema_slow and ema_fast > ema_slow
            )
            # Bearish crossover: fast crosses below slow
            bear_cross = (
                self._prev_ema_fast >= self._prev_ema_slow and ema_fast < ema_slow
            )

            if bull_cross and regime.total_score > 0:
                self._enter_trade(OrderSide.BUY, price, atr, "TREND_BULL")
                return
            elif bear_cross and regime.total_score < 0:
                self._enter_trade(OrderSide.SELL, price, atr, "TREND_BEAR")
                return

        # MEAN REVERSION: BB bounce + RSI confirmation
        # Allow in ranging market OR weak trends (|score| < 2.0)
        if not regime.is_trending and regime.confidence >= config.confidence_threshold:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_percent = (price - bb_lower) / bb_range

                # Buy at lower band
                if bb_percent <= 0.15 and rsi < config.mr_rsi_buy_threshold:
                    self._enter_trade(OrderSide.BUY, price, atr, "MR_BUY")
                    return

                # Sell at upper band
                if bb_percent >= 0.85 and rsi > config.mr_rsi_sell_threshold:
                    self._enter_trade(OrderSide.SELL, price, atr, "MR_SELL")
                    return

    def _enter_trade(
        self,
        side: OrderSide,
        price: float,
        atr: float,
        reason: str,
    ) -> None:
        """Submit a market order entry."""
        config: OlympexStrategyConfig = self.config

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(config.position_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        # Track entry state
        self._entry_price = price
        self._entry_side = side
        self._entry_bar_count = self._bar_count
        self._entry_regime_trending = self._current_regime.is_trending if self._current_regime else False

        # Update stats
        if reason.startswith("TREND"):
            self.stats["trend_entries"] += 1
        else:
            self.stats["mr_entries"] += 1

        self.log.info(f"ENTRY: {reason} | {side.name} @ {price:.2f} | ATR={atr:.2f} | Regime={self._current_regime.regime.value if self._current_regime else 'N/A'}")

    def _check_exits(self, price: float, atr: float, regime: RegimeState) -> None:
        """Check exit conditions for open position."""
        if self._entry_price is None or self._entry_side is None:
            return

        config: OlympexStrategyConfig = self.config
        bars_held = self._bar_count - self._entry_bar_count
        sl_distance = atr * config.sl_atr_multiplier
        tp_distance = sl_distance * config.rr_ratio

        if self._entry_side == OrderSide.BUY:
            pnl = price - self._entry_price
            sl_hit = price <= self._entry_price - sl_distance
            tp_hit = price >= self._entry_price + tp_distance

            # Trailing stop (activate after 50% of TP reached)
            trailing_distance = atr * config.trailing_atr_multiplier
            # Simple trailing: if we've gained > trailing_distance, check if price dropped back
            trailing_hit = (
                pnl > trailing_distance and price < self._entry_price + pnl - trailing_distance
            )
        else:  # SELL
            pnl = self._entry_price - price
            sl_hit = price >= self._entry_price + sl_distance
            tp_hit = price <= self._entry_price - tp_distance

            trailing_distance = atr * config.trailing_atr_multiplier
            trailing_hit = (
                pnl > trailing_distance and price > self._entry_price - pnl + trailing_distance
            )

        # Regime flip: exit trend trades when regime reverses
        regime_flip = False
        if self._entry_regime_trending:
            if self._entry_side == OrderSide.BUY and regime.total_score < -1.0:
                regime_flip = True
            elif self._entry_side == OrderSide.SELL and regime.total_score > 1.0:
                regime_flip = True

        # Timeout
        timeout = bars_held >= config.max_holding_bars

        # Exit logic — priority order
        exit_reason = None
        if sl_hit:
            exit_reason = "STOP_LOSS"
            self.stats["exits_sl"] += 1
        elif tp_hit:
            exit_reason = "TAKE_PROFIT"
            self.stats["exits_tp"] += 1
        elif trailing_hit:
            exit_reason = "TRAILING_STOP"
            self.stats["exits_trailing"] += 1
        elif regime_flip:
            exit_reason = "REGIME_FLIP"
            self.stats["exits_regime_flip"] += 1
        elif timeout:
            exit_reason = "TIMEOUT"
            self.stats["exits_timeout"] += 1

        if exit_reason:
            self._exit_trade(price, pnl, exit_reason)

    def _exit_trade(self, price: float, pnl: float, reason: str) -> None:
        """Close current position."""
        if self._entry_side == OrderSide.BUY:
            close_side = OrderSide.SELL
        else:
            close_side = OrderSide.BUY

        config: OlympexStrategyConfig = self.config
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=close_side,
            quantity=self.instrument.make_qty(config.position_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        self.log.info(f"EXIT: {reason} | {close_side.name} @ {price:.2f} | PnL={pnl:.2f}")

        # Reset state
        self._entry_price = None
        self._entry_side = None
        self._entry_bar_count = 0
        self._entry_regime_trending = False

    def on_stop(self) -> None:
        """Log strategy statistics on stop."""
        self.log.info("=" * 50)
        self.log.info("OLYMPEX STRATEGY STATISTICS")
        self.log.info("=" * 50)
        self.log.info(f"Total bars processed: {self._bar_count}")
        self.log.info(f"Trend entries: {self.stats['trend_entries']}")
        self.log.info(f"Mean reversion entries: {self.stats['mr_entries']}")
        self.log.info(f"Exits - TP: {self.stats['exits_tp']}")
        self.log.info(f"Exits - SL: {self.stats['exits_sl']}")
        self.log.info(f"Exits - Trailing: {self.stats['exits_trailing']}")
        self.log.info(f"Exits - Timeout: {self.stats['exits_timeout']}")
        self.log.info(f"Exits - Regime Flip: {self.stats['exits_regime_flip']}")
        self.log.info("=" * 50)
