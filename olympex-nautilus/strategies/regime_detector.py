"""
Market Regime Detector — Multi-indicator scoring system.

Classifies market conditions into regimes (trending/ranging) using:
- RSI (momentum)
- MACD (trend direction + strength)
- Bollinger Bands %B (price position within bands)
- EMA Alignment (trend structure)
- ATR (volatility weighting)
"""

from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    WEAK_BULL = "weak_bull"
    NEUTRAL = "neutral"
    WEAK_BEAR = "weak_bear"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


@dataclass
class RegimeState:
    """Current market regime assessment."""

    total_score: float
    regime: Regime
    confidence: float  # 0.0 to 1.0
    is_trending: bool
    is_ranging: bool
    volatility: str  # "low", "normal", "high"

    # Individual component scores
    rsi_score: float = 0.0
    macd_score: float = 0.0
    bb_score: float = 0.0
    ema_score: float = 0.0


MAX_POSSIBLE_SCORE = 7.0  # 2.0 + 2.0 + 1.5 + 1.5


class MarketRegimeDetector:
    """
    Scores market regime using multiple indicators.

    The detector does NOT own indicators — it receives indicator values
    from the strategy and computes regime scores.
    """

    def __init__(self, atr_high_percentile: float = 1.5, atr_low_percentile: float = 0.5):
        self.atr_high_mult = atr_high_percentile
        self.atr_low_mult = atr_low_percentile
        self._atr_history: list[float] = []
        self._atr_window = 100
        self._prev_macd: float | None = None

    def compute(
        self,
        rsi_value: float,
        macd_value: float,
        bb_upper: float,
        bb_lower: float,
        bb_middle: float,
        price: float,
        ema_fast: float,
        ema_slow: float,
        atr_value: float,
    ) -> RegimeState:
        """Compute regime state from current indicator values."""

        # 1. RSI Score (-2 to +2)
        rsi_score = self._score_rsi(rsi_value)

        # 2. MACD Score (-2 to +2)
        macd_score = self._score_macd(macd_value)

        # 3. Bollinger Bands %B Score (-1.5 to +1.5)
        bb_score = self._score_bb(price, bb_upper, bb_lower, bb_middle)

        # 4. EMA Alignment Score (-1.5 to +1.5)
        ema_score = self._score_ema(price, ema_fast, ema_slow)

        # 5. Volatility assessment
        volatility = self._assess_volatility(atr_value)

        # Total score
        total_score = rsi_score + macd_score + bb_score + ema_score

        # Classify regime
        abs_score = abs(total_score)
        if abs_score >= 4.0:
            regime = Regime.STRONG_BULL if total_score > 0 else Regime.STRONG_BEAR
        elif abs_score >= 2.0:
            regime = Regime.BULL if total_score > 0 else Regime.BEAR
        elif abs_score >= 0.5:
            regime = Regime.WEAK_BULL if total_score > 0 else Regime.WEAK_BEAR
        else:
            regime = Regime.NEUTRAL

        confidence = abs_score / MAX_POSSIBLE_SCORE

        # Trending: |score| >= 2.0
        # Ranging: |score| < 1.0 (widened from 0.5 to fix mean reversion)
        is_trending = abs_score >= 2.0
        is_ranging = abs_score < 1.0

        # Update MACD history
        self._prev_macd = macd_value

        return RegimeState(
            total_score=total_score,
            regime=regime,
            confidence=confidence,
            is_trending=is_trending,
            is_ranging=is_ranging,
            volatility=volatility,
            rsi_score=rsi_score,
            macd_score=macd_score,
            bb_score=bb_score,
            ema_score=ema_score,
        )

    def _score_rsi(self, rsi: float) -> float:
        """RSI score: -2 (oversold/bearish) to +2 (overbought/bullish)."""
        if rsi <= 30:
            return -2.0
        elif rsi >= 70:
            return 2.0
        elif rsi < 50:
            # Interpolate -2 to 0 for RSI 30-50
            return -2.0 * (50 - rsi) / 20
        else:
            # Interpolate 0 to +2 for RSI 50-70
            return 2.0 * (rsi - 50) / 20

    def _score_macd(self, macd: float) -> float:
        """MACD score based on value and direction."""
        # Direction component
        if self._prev_macd is not None:
            rising = macd > self._prev_macd
        else:
            rising = macd > 0

        # Magnitude — use sign and direction
        if macd > 0 and rising:
            score = 2.0
        elif macd > 0 and not rising:
            score = 0.5
        elif macd < 0 and not rising:
            score = -2.0
        elif macd < 0 and rising:
            score = -0.5
        else:
            score = 0.0

        return score

    def _score_bb(self, price: float, upper: float, lower: float, middle: float) -> float:
        """Bollinger Band %B score: -1.5 (at lower) to +1.5 (at upper)."""
        band_width = upper - lower
        if band_width <= 0:
            return 0.0

        percent_b = (price - lower) / band_width  # 0 = lower band, 1 = upper band

        if percent_b <= 0.2:
            return -1.5
        elif percent_b >= 0.8:
            return 1.5
        elif percent_b < 0.5:
            # Interpolate -1.5 to 0 for %B 0.2 to 0.5
            return -1.5 * (0.5 - percent_b) / 0.3
        else:
            # Interpolate 0 to 1.5 for %B 0.5 to 0.8
            return 1.5 * (percent_b - 0.5) / 0.3

    def _score_ema(self, price: float, ema_fast: float, ema_slow: float) -> float:
        """EMA alignment score: price vs fast vs slow."""
        if price > ema_fast > ema_slow:
            return 1.5  # Strong bullish alignment
        elif price > ema_fast and ema_fast <= ema_slow:
            return 0.5  # Price above fast but EMAs not aligned
        elif price < ema_fast < ema_slow:
            return -1.5  # Strong bearish alignment
        elif price < ema_fast and ema_fast >= ema_slow:
            return -0.5  # Price below fast but EMAs not aligned
        else:
            return 0.0

    def _assess_volatility(self, atr_value: float) -> str:
        """Assess current volatility level relative to recent history."""
        self._atr_history.append(atr_value)
        if len(self._atr_history) > self._atr_window:
            self._atr_history = self._atr_history[-self._atr_window :]

        if len(self._atr_history) < 10:
            return "normal"

        avg_atr = sum(self._atr_history) / len(self._atr_history)
        if avg_atr == 0:
            return "normal"

        ratio = atr_value / avg_atr
        if ratio > self.atr_high_mult:
            return "high"
        elif ratio < self.atr_low_mult:
            return "low"
        return "normal"
