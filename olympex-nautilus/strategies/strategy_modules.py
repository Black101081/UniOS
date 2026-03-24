"""
Strategy Modules — Each module implements a specific trading approach.

Each module has:
  - check_entry(): Returns (side, reason) or None
  - get_sl_tp(): Returns (sl_distance, tp_distance) based on ATR

The modules do NOT submit orders — they return signals.
The master strategy handles all order management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nautilus_trader.model.enums import OrderSide

from .market_measure import MarketState
from .regime_classifier import Direction, RegimeSignal


class SignalType(Enum):
    TREND = "trend"
    MEAN_REVERSION = "mr"
    BREAKOUT = "breakout"
    SCALP = "scalp"
    FADE = "fade"


@dataclass
class TradeSignal:
    """A trade signal from a strategy module."""
    side: OrderSide
    signal_type: SignalType
    reason: str
    sl_atr_mult: float     # stop-loss distance as ATR multiplier
    tp_atr_mult: float     # take-profit distance as ATR multiplier
    strength: float        # 0-1, signal strength/conviction


# =============================================================================
# TREND FOLLOWING MODULE
# =============================================================================
class TrendModule:
    """
    Trend following: ride established trends.
    Entry: EMA crossover confirmed by ADX and momentum.
    """

    def __init__(
        self,
        adx_min: float = 20.0,
        rsi_bull_min: float = 45.0,
        rsi_bear_max: float = 55.0,
        sl_atr: float = 2.0,
        rr_ratio: float = 2.5,
    ):
        self.adx_min = adx_min
        self.rsi_bull_min = rsi_bull_min
        self.rsi_bear_max = rsi_bear_max
        self.sl_atr = sl_atr
        self.rr_ratio = rr_ratio

        self._prev_ema_fast: float | None = None
        self._prev_ema_slow: float | None = None

    def check_entry(
        self, state: MarketState, signal: RegimeSignal,
        ema_fast: float, ema_slow: float,
    ) -> TradeSignal | None:
        if self._prev_ema_fast is None:
            self._update(ema_fast, ema_slow)
            return None

        # Need confirmed trend
        if state.adx_value < self.adx_min:
            self._update(ema_fast, ema_slow)
            return None

        bull_cross = self._prev_ema_fast <= self._prev_ema_slow and ema_fast > ema_slow
        bear_cross = self._prev_ema_fast >= self._prev_ema_slow and ema_fast < ema_slow

        # Bullish trend entry
        if bull_cross and signal.direction == Direction.BULLISH:
            if state.rsi > self.rsi_bull_min and state.macd_histogram > 0:
                strength = signal.trend_score * min(state.adx_value / 40, 1.0)
                self._update(ema_fast, ema_slow)
                return TradeSignal(
                    side=OrderSide.BUY,
                    signal_type=SignalType.TREND,
                    reason=f"TREND_BULL ADX={state.adx_value:.0f}",
                    sl_atr_mult=self.sl_atr,
                    tp_atr_mult=self.sl_atr * self.rr_ratio,
                    strength=strength,
                )

        # Bearish trend entry
        if bear_cross and signal.direction == Direction.BEARISH:
            if state.rsi < self.rsi_bear_max and state.macd_histogram < 0:
                strength = signal.trend_score * min(state.adx_value / 40, 1.0)
                self._update(ema_fast, ema_slow)
                return TradeSignal(
                    side=OrderSide.SELL,
                    signal_type=SignalType.TREND,
                    reason=f"TREND_BEAR ADX={state.adx_value:.0f}",
                    sl_atr_mult=self.sl_atr,
                    tp_atr_mult=self.sl_atr * self.rr_ratio,
                    strength=strength,
                )

        self._update(ema_fast, ema_slow)
        return None

    def _update(self, ema_fast: float, ema_slow: float) -> None:
        self._prev_ema_fast = ema_fast
        self._prev_ema_slow = ema_slow

    def reset(self) -> None:
        self._prev_ema_fast = None
        self._prev_ema_slow = None


# =============================================================================
# MEAN REVERSION MODULE
# =============================================================================
class MeanReversionModule:
    """
    Mean reversion: fade overextensions in ranging markets.
    Entry: Price at BB extreme + RSI confirmation.
    """

    def __init__(
        self,
        bb_threshold: float = 0.10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sl_atr: float = 1.5,
        rr_ratio: float = 1.5,
    ):
        self.bb_threshold = bb_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sl_atr = sl_atr
        self.rr_ratio = rr_ratio

    def check_entry(
        self, state: MarketState, signal: RegimeSignal,
        price: float, bb_upper: float, bb_lower: float,
    ) -> TradeSignal | None:
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return None

        bb_pct = (price - bb_lower) / bb_range

        # Oversold bounce
        if bb_pct <= self.bb_threshold and state.rsi < self.rsi_oversold:
            strength = signal.mr_score * (1 - bb_pct / self.bb_threshold)
            return TradeSignal(
                side=OrderSide.BUY,
                signal_type=SignalType.MEAN_REVERSION,
                reason=f"MR_BUY BB%={bb_pct:.2f} RSI={state.rsi:.0f}",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        # Overbought fade
        if bb_pct >= (1 - self.bb_threshold) and state.rsi > self.rsi_overbought:
            strength = signal.mr_score * (bb_pct - (1 - self.bb_threshold)) / self.bb_threshold
            return TradeSignal(
                side=OrderSide.SELL,
                signal_type=SignalType.MEAN_REVERSION,
                reason=f"MR_SELL BB%={bb_pct:.2f} RSI={state.rsi:.0f}",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        return None

    def reset(self) -> None:
        pass


# =============================================================================
# BREAKOUT MODULE
# =============================================================================
class BreakoutModule:
    """
    Breakout: enter on range expansion with volume confirmation.
    Entry: Price breaks N-bar high/low + volume surge.
    """

    def __init__(
        self,
        lookback: int = 20,
        volume_mult: float = 1.5,
        sl_atr: float = 1.5,
        rr_ratio: float = 3.0,
    ):
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.sl_atr = sl_atr
        self.rr_ratio = rr_ratio

        self._high_history: list[float] = []
        self._low_history: list[float] = []

    def check_entry(
        self, state: MarketState, signal: RegimeSignal,
        price: float, high: float, low: float,
    ) -> TradeSignal | None:
        self._high_history.append(high)
        self._low_history.append(low)

        if len(self._high_history) > self.lookback + 1:
            self._high_history = self._high_history[-(self.lookback + 1):]
            self._low_history = self._low_history[-(self.lookback + 1):]

        if len(self._high_history) <= self.lookback:
            return None

        # Previous N-bar range (excluding current bar)
        prev_highs = self._high_history[:-1]
        prev_lows = self._low_history[:-1]
        range_high = max(prev_highs[-self.lookback:])
        range_low = min(prev_lows[-self.lookback:])

        has_volume = state.volume_ratio >= self.volume_mult

        # Bullish breakout
        if price > range_high and has_volume:
            strength = signal.breakout_score * min(state.volume_ratio / 2, 1.0)
            return TradeSignal(
                side=OrderSide.BUY,
                signal_type=SignalType.BREAKOUT,
                reason=f"BREAK_UP vol={state.volume_ratio:.1f}x",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        # Bearish breakout
        if price < range_low and has_volume:
            strength = signal.breakout_score * min(state.volume_ratio / 2, 1.0)
            return TradeSignal(
                side=OrderSide.SELL,
                signal_type=SignalType.BREAKOUT,
                reason=f"BREAK_DN vol={state.volume_ratio:.1f}x",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        return None

    def reset(self) -> None:
        self._high_history.clear()
        self._low_history.clear()


# =============================================================================
# SCALP MODULE
# =============================================================================
class ScalpModule:
    """
    Scalp: quick in-out trades in high-volatility environments.
    Entry: Strong wick rejection + volume.
    """

    def __init__(
        self,
        min_wick_ratio: float = 0.4,
        min_volume_ratio: float = 1.2,
        sl_atr: float = 0.8,
        rr_ratio: float = 1.5,
    ):
        self.min_wick_ratio = min_wick_ratio
        self.min_volume_ratio = min_volume_ratio
        self.sl_atr = sl_atr
        self.rr_ratio = rr_ratio

    def check_entry(
        self, state: MarketState, signal: RegimeSignal,
        price: float, open_: float,
    ) -> TradeSignal | None:
        if state.volume_ratio < self.min_volume_ratio:
            return None

        # Long lower wick = buying pressure (scalp long)
        if state.lower_wick_ratio > self.min_wick_ratio and price > open_:
            strength = signal.scalp_score * state.lower_wick_ratio
            return TradeSignal(
                side=OrderSide.BUY,
                signal_type=SignalType.SCALP,
                reason=f"SCALP_BUY wick={state.lower_wick_ratio:.2f}",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        # Long upper wick = selling pressure (scalp short)
        if state.upper_wick_ratio > self.min_wick_ratio and price < open_:
            strength = signal.scalp_score * state.upper_wick_ratio
            return TradeSignal(
                side=OrderSide.SELL,
                signal_type=SignalType.SCALP,
                reason=f"SCALP_SELL wick={state.upper_wick_ratio:.2f}",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        return None

    def reset(self) -> None:
        pass


# =============================================================================
# FADE MODULE
# =============================================================================
class FadeModule:
    """
    Fade: counter-trade extreme momentum (3-sigma moves).
    Entry: After extreme candle, bet on partial reversion.
    """

    def __init__(
        self,
        roc_threshold: float = 0.3,   # % move to consider "extreme"
        rsi_extreme_low: float = 20.0,
        rsi_extreme_high: float = 80.0,
        sl_atr: float = 2.0,
        rr_ratio: float = 1.0,
    ):
        self.roc_threshold = roc_threshold
        self.rsi_extreme_low = rsi_extreme_low
        self.rsi_extreme_high = rsi_extreme_high
        self.sl_atr = sl_atr
        self.rr_ratio = rr_ratio

    def check_entry(
        self, state: MarketState, signal: RegimeSignal,
    ) -> TradeSignal | None:
        # Fade extreme down move
        if state.roc < -self.roc_threshold and state.rsi < self.rsi_extreme_low:
            strength = signal.fade_score * min(abs(state.roc) / 0.5, 1.0)
            return TradeSignal(
                side=OrderSide.BUY,
                signal_type=SignalType.FADE,
                reason=f"FADE_BUY ROC={state.roc:.2f}% RSI={state.rsi:.0f}",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        # Fade extreme up move
        if state.roc > self.roc_threshold and state.rsi > self.rsi_extreme_high:
            strength = signal.fade_score * min(abs(state.roc) / 0.5, 1.0)
            return TradeSignal(
                side=OrderSide.SELL,
                signal_type=SignalType.FADE,
                reason=f"FADE_SELL ROC={state.roc:.2f}% RSI={state.rsi:.0f}",
                sl_atr_mult=self.sl_atr,
                tp_atr_mult=self.sl_atr * self.rr_ratio,
                strength=strength,
            )

        return None

    def reset(self) -> None:
        pass
