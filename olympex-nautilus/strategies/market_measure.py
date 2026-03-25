"""
Market Measurement Engine — Multi-dimensional market state quantification.

Measures 5 dimensions on every bar:
  1. Volatility: ATR regime, BB width, realized vol percentile
  2. Trend: ADX, EMA alignment, linear regression R², Hurst estimate
  3. Momentum: RSI, MACD histogram, Rate of Change
  4. Volume: Relative volume, volume trend, buy/sell pressure
  5. Microstructure: Candle body ratio, wick ratio

All measurements are normalized to comparable scales (0-1 or -1 to +1)
so they can be combined with dynamic weights.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MarketState:
    """Complete market measurement snapshot."""

    # Volatility
    atr_ratio: float = 0.0       # current ATR / avg ATR — >1 = high vol
    bb_width: float = 0.0        # normalized BB width
    vol_percentile: float = 0.5  # 0-1, where current vol sits in history

    # Trend
    adx_value: float = 0.0       # 0-100, trend strength
    ema_alignment: float = 0.0   # -1 (bear) to +1 (bull) alignment
    trend_r2: float = 0.0        # 0-1, trend quality (linear reg R²)
    hurst: float = 0.5           # <0.5 mean-revert, =0.5 random, >0.5 trending

    # Momentum
    rsi: float = 50.0
    macd_histogram: float = 0.0  # normalized
    roc: float = 0.0             # rate of change (%)

    # Volume
    volume_ratio: float = 1.0    # current / avg — >1 = high volume
    volume_trend: float = 1.0    # short-term vol avg / long-term vol avg
    buy_pressure: float = 0.5    # 0 = all sell, 1 = all buy

    # Microstructure
    body_ratio: float = 0.5      # |close-open| / (high-low)
    upper_wick_ratio: float = 0.0
    lower_wick_ratio: float = 0.0

    # Meta
    bar_count: int = 0
    is_valid: bool = False       # True when all warmup complete


class MarketMeasure:
    """
    Computes multi-dimensional market measurements from raw indicator values.

    This class does NOT own NautilusTrader indicators.
    It receives values from the strategy and computes derived metrics.
    """

    def __init__(
        self,
        atr_lookback: int = 100,
        vol_lookback: int = 200,
        hurst_lookback: int = 100,
        adx_period: int = 14,
        linreg_period: int = 30,
    ):
        self._atr_lookback = atr_lookback
        self._vol_lookback = vol_lookback
        self._hurst_lookback = hurst_lookback
        self._adx_period = adx_period
        self._linreg_period = linreg_period

        # History buffers
        self._atr_history: deque[float] = deque(maxlen=atr_lookback)
        self._close_history: deque[float] = deque(maxlen=max(vol_lookback, hurst_lookback))
        self._volume_history: deque[float] = deque(maxlen=vol_lookback)
        self._tr_history: deque[float] = deque(maxlen=vol_lookback)

        # ADX computation state
        self._plus_dm_ema: float = 0.0
        self._minus_dm_ema: float = 0.0
        self._tr_ema: float = 0.0
        self._dx_history: deque[float] = deque(maxlen=adx_period)
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None

        self._bar_count: int = 0

    def update(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        atr_value: float,
        rsi_value: float,
        macd_value: float,
        macd_signal: float,
        ema_fast: float,
        ema_slow: float,
        bb_upper: float,
        bb_lower: float,
        bb_middle: float,
    ) -> MarketState:
        """Update all measurements with new bar data and return current state."""
        self._bar_count += 1

        # Store history
        self._atr_history.append(atr_value)
        self._close_history.append(close)
        self._volume_history.append(volume)

        # True Range
        if self._prev_close is not None:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr_history.append(tr)

        # --- VOLATILITY ---
        atr_ratio = self._compute_atr_ratio(atr_value)
        bb_width = self._compute_bb_width(bb_upper, bb_lower, bb_middle)
        vol_percentile = self._compute_vol_percentile(atr_value)

        # --- TREND ---
        adx = self._compute_adx(high, low, close)
        ema_alignment = self._compute_ema_alignment(close, ema_fast, ema_slow)
        trend_r2 = self._compute_trend_r2()
        hurst = self._compute_hurst()

        # --- MOMENTUM ---
        macd_histogram = macd_value - macd_signal
        roc = self._compute_roc(close, 14)

        # --- VOLUME ---
        volume_ratio = self._compute_volume_ratio(volume, 20)
        volume_trend = self._compute_volume_trend()
        buy_pressure = self._compute_buy_pressure(open_, high, low, close)

        # --- MICROSTRUCTURE ---
        body_ratio, upper_wick, lower_wick = self._compute_candle_structure(
            open_, high, low, close
        )

        # Update prev bar
        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        is_valid = self._bar_count >= max(self._adx_period * 2, self._linreg_period, 50)

        return MarketState(
            atr_ratio=atr_ratio,
            bb_width=bb_width,
            vol_percentile=vol_percentile,
            adx_value=adx,
            ema_alignment=ema_alignment,
            trend_r2=trend_r2,
            hurst=hurst,
            rsi=rsi_value,
            macd_histogram=macd_histogram,
            roc=roc,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            buy_pressure=buy_pressure,
            body_ratio=body_ratio,
            upper_wick_ratio=upper_wick,
            lower_wick_ratio=lower_wick,
            bar_count=self._bar_count,
            is_valid=is_valid,
        )

    def reset(self) -> None:
        """Reset all state (call on data gaps)."""
        self._atr_history.clear()
        self._close_history.clear()
        self._volume_history.clear()
        self._tr_history.clear()
        self._dx_history.clear()
        self._plus_dm_ema = 0.0
        self._minus_dm_ema = 0.0
        self._tr_ema = 0.0
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._bar_count = 0

    # -------------------------------------------------------------------------
    # Volatility computations
    # -------------------------------------------------------------------------
    def _compute_atr_ratio(self, atr: float) -> float:
        if len(self._atr_history) < 20:
            return 1.0
        avg = sum(self._atr_history) / len(self._atr_history)
        return atr / avg if avg > 0 else 1.0

    def _compute_bb_width(self, upper: float, lower: float, middle: float) -> float:
        if middle <= 0:
            return 0.0
        return (upper - lower) / middle

    def _compute_vol_percentile(self, atr: float) -> float:
        if len(self._atr_history) < 20:
            return 0.5
        sorted_atr = sorted(self._atr_history)
        rank = sum(1 for x in sorted_atr if x <= atr)
        return rank / len(sorted_atr)

    # -------------------------------------------------------------------------
    # Trend computations
    # -------------------------------------------------------------------------
    def _compute_adx(self, high: float, low: float, close: float) -> float:
        """Simplified ADX computation."""
        if self._prev_high is None:
            return 0.0

        # Directional movement
        up_move = high - self._prev_high
        down_move = self._prev_low - low

        plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
        minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0

        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))

        # EMA smoothing
        period = self._adx_period
        alpha = 1.0 / period

        if self._bar_count <= period + 1:
            # SMA accumulation phase
            self._plus_dm_ema += plus_dm
            self._minus_dm_ema += minus_dm
            self._tr_ema += tr
            if self._bar_count == period + 1:
                self._plus_dm_ema /= period
                self._minus_dm_ema /= period
                self._tr_ema /= period
            return 0.0

        self._plus_dm_ema = self._plus_dm_ema * (1 - alpha) + plus_dm * alpha
        self._minus_dm_ema = self._minus_dm_ema * (1 - alpha) + minus_dm * alpha
        self._tr_ema = self._tr_ema * (1 - alpha) + tr * alpha

        if self._tr_ema <= 0:
            return 0.0

        plus_di = 100 * self._plus_dm_ema / self._tr_ema
        minus_di = 100 * self._minus_dm_ema / self._tr_ema

        di_sum = plus_di + minus_di
        if di_sum <= 0:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / di_sum
        self._dx_history.append(dx)

        if len(self._dx_history) < self._adx_period:
            return 0.0

        return sum(self._dx_history) / len(self._dx_history)

    def _compute_ema_alignment(self, price: float, ema_fast: float, ema_slow: float) -> float:
        """EMA alignment score: -1 (strong bear) to +1 (strong bull)."""
        if ema_slow <= 0:
            return 0.0

        # Price vs EMAs alignment
        price_vs_fast = (price - ema_fast) / ema_slow
        fast_vs_slow = (ema_fast - ema_slow) / ema_slow

        # Both positive = bullish, both negative = bearish
        alignment = price_vs_fast + fast_vs_slow

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, alignment * 50))  # scale factor

    def _compute_trend_r2(self) -> float:
        """R² of linear regression on recent closes."""
        n = min(self._linreg_period, len(self._close_history))
        if n < 10:
            return 0.0

        closes = list(self._close_history)[-n:]
        mean_y = sum(closes) / n
        mean_x = (n - 1) / 2.0

        ss_xx = sum((i - mean_x) ** 2 for i in range(n))
        ss_yy = sum((y - mean_y) ** 2 for y in closes)
        ss_xy = sum((i - mean_x) * (closes[i] - mean_y) for i in range(n))

        if ss_xx <= 0 or ss_yy <= 0:
            return 0.0

        r = ss_xy / math.sqrt(ss_xx * ss_yy)
        return r * r  # R²

    def _compute_hurst(self) -> float:
        """Simplified Hurst exponent via rescaled range method."""
        n = min(self._hurst_lookback, len(self._close_history))
        if n < 20:
            return 0.5

        closes = list(self._close_history)[-n:]
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]

        if len(returns) < 10:
            return 0.5

        # R/S analysis with two partition sizes
        rs_values = []
        for chunk_size in [10, 20, 40]:
            if chunk_size > len(returns):
                continue
            chunks = [returns[i:i + chunk_size] for i in range(0, len(returns) - chunk_size + 1, chunk_size)]
            for chunk in chunks:
                mean_r = sum(chunk) / len(chunk)
                deviations = [r - mean_r for r in chunk]
                cumdev = []
                s = 0
                for d in deviations:
                    s += d
                    cumdev.append(s)
                R = max(cumdev) - min(cumdev)
                S = math.sqrt(sum(d * d for d in deviations) / len(deviations))
                if S > 0:
                    rs_values.append((math.log(chunk_size), math.log(R / S)))

        if len(rs_values) < 3:
            return 0.5

        # Linear regression of log(R/S) vs log(n)
        xs = [v[0] for v in rs_values]
        ys = [v[1] for v in rs_values]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        ss_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
        ss_xx = sum((x - mean_x) ** 2 for x in xs)

        if ss_xx <= 0:
            return 0.5

        hurst = ss_xy / ss_xx
        return max(0.0, min(1.0, hurst))

    # -------------------------------------------------------------------------
    # Momentum computations
    # -------------------------------------------------------------------------
    def _compute_roc(self, close: float, period: int) -> float:
        if len(self._close_history) <= period:
            return 0.0
        prev = list(self._close_history)[-period - 1]
        if prev <= 0:
            return 0.0
        return (close - prev) / prev * 100

    # -------------------------------------------------------------------------
    # Volume computations
    # -------------------------------------------------------------------------
    def _compute_volume_ratio(self, volume: float, period: int) -> float:
        if len(self._volume_history) < period:
            return 1.0
        recent = list(self._volume_history)[-period:]
        avg = sum(recent) / len(recent)
        return volume / avg if avg > 0 else 1.0

    def _compute_volume_trend(self) -> float:
        if len(self._volume_history) < 20:
            return 1.0
        vols = list(self._volume_history)
        short = sum(vols[-5:]) / 5
        long_ = sum(vols[-20:]) / 20
        return short / long_ if long_ > 0 else 1.0

    def _compute_buy_pressure(self, o: float, h: float, l: float, c: float) -> float:
        """Estimate buy pressure from candle shape (0=all sell, 1=all buy)."""
        bar_range = h - l
        if bar_range <= 0:
            return 0.5
        # Close position in bar range
        return (c - l) / bar_range

    # -------------------------------------------------------------------------
    # Microstructure computations
    # -------------------------------------------------------------------------
    def _compute_candle_structure(
        self, o: float, h: float, l: float, c: float
    ) -> tuple[float, float, float]:
        bar_range = h - l
        if bar_range <= 0:
            return 0.0, 0.0, 0.0

        body = abs(c - o)
        body_ratio = body / bar_range

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        upper_ratio = upper_wick / bar_range
        lower_ratio = lower_wick / bar_range

        return body_ratio, upper_ratio, lower_ratio
