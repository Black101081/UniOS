"""
Market Regime Classifier — Classifies market state into actionable regimes.

Takes MarketState measurements and produces a RegimeSignal with:
  - Primary regime (trending, ranging, breakout, squeeze, volatile)
  - Direction (bullish, bearish, neutral)
  - Confidence score (0-1)
  - Per-strategy activation scores

Key design principle: weights of each measurement dimension
change depending on the current regime (dynamic weighting).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .market_measure import MarketState


class Regime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    SQUEEZE = "squeeze"
    VOLATILE = "volatile"


class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class RegimeSignal:
    """Output of the regime classifier."""

    regime: Regime
    direction: Direction
    confidence: float              # 0-1

    # Per-strategy activation scores (0-1, how suitable is each strategy now)
    trend_score: float = 0.0
    mr_score: float = 0.0
    breakout_score: float = 0.0
    scalp_score: float = 0.0
    fade_score: float = 0.0

    # Raw component scores for debugging
    volatility_score: float = 0.0  # low=0, high=1
    trend_strength: float = 0.0    # 0=no trend, 1=strong trend
    momentum_score: float = 0.0    # -1=bearish, +1=bullish
    volume_score: float = 0.0      # 0=low, 1=high
    structure_score: float = 0.0   # 0=indecisive, 1=decisive candles


class RegimeClassifier:
    """
    Classifies MarketState into regimes using adaptive thresholds.

    The classifier maintains a recent history of regime signals
    to provide smooth transitions and avoid whipsawing.
    """

    def __init__(
        self,
        # ADX thresholds
        adx_trending: float = 25.0,
        adx_weak: float = 15.0,
        # Volatility thresholds
        vol_high: float = 1.5,       # ATR ratio > 1.5 = high vol
        vol_low: float = 0.6,        # ATR ratio < 0.6 = low vol (squeeze)
        # BB width thresholds
        bb_squeeze: float = 0.01,    # very narrow bands
        bb_wide: float = 0.04,       # wide bands
        # Hurst thresholds
        hurst_trending: float = 0.55,
        hurst_mean_revert: float = 0.45,
        # Smoothing
        regime_persistence: int = 3,  # bars to confirm regime change
    ):
        self._adx_trending = adx_trending
        self._adx_weak = adx_weak
        self._vol_high = vol_high
        self._vol_low = vol_low
        self._bb_squeeze = bb_squeeze
        self._bb_wide = bb_wide
        self._hurst_trending = hurst_trending
        self._hurst_mean_revert = hurst_mean_revert
        self._regime_persistence = regime_persistence

        # State
        self._prev_regime: Regime = Regime.RANGING
        self._regime_count: int = 0  # how many bars current regime has been active
        self._pending_regime: Regime | None = None
        self._pending_count: int = 0

    def classify(self, state: MarketState) -> RegimeSignal:
        """Classify the current market state into a regime."""
        if not state.is_valid:
            return RegimeSignal(
                regime=Regime.RANGING,
                direction=Direction.NEUTRAL,
                confidence=0.0,
            )

        # Step 1: Compute component scores
        vol_score = self._score_volatility(state)
        trend_str = self._score_trend_strength(state)
        mom_score = self._score_momentum(state)
        vol_act_score = self._score_volume(state)
        struct_score = self._score_structure(state)

        # Step 2: Determine raw regime
        raw_regime = self._determine_regime(state, vol_score, trend_str)

        # Step 3: Apply persistence filter (avoid whipsawing)
        regime = self._apply_persistence(raw_regime)

        # Step 4: Determine direction
        direction = self._determine_direction(state, mom_score)

        # Step 5: Compute confidence
        confidence = self._compute_confidence(state, regime, vol_score, trend_str)

        # Step 6: Compute per-strategy activation scores
        trend_act = self._trend_activation(state, regime, trend_str, vol_act_score)
        mr_act = self._mr_activation(state, regime, trend_str, vol_score)
        breakout_act = self._breakout_activation(state, regime, vol_score, vol_act_score)
        scalp_act = self._scalp_activation(state, regime, vol_score, vol_act_score)
        fade_act = self._fade_activation(state, regime, vol_score, mom_score)

        return RegimeSignal(
            regime=regime,
            direction=direction,
            confidence=confidence,
            trend_score=trend_act,
            mr_score=mr_act,
            breakout_score=breakout_act,
            scalp_score=scalp_act,
            fade_score=fade_act,
            volatility_score=vol_score,
            trend_strength=trend_str,
            momentum_score=mom_score,
            volume_score=vol_act_score,
            structure_score=struct_score,
        )

    def reset(self) -> None:
        self._prev_regime = Regime.RANGING
        self._regime_count = 0
        self._pending_regime = None
        self._pending_count = 0

    # -------------------------------------------------------------------------
    # Component scoring (all normalized 0-1 or -1 to +1)
    # -------------------------------------------------------------------------
    def _score_volatility(self, s: MarketState) -> float:
        """0 = very low vol, 1 = very high vol."""
        # Combine ATR ratio and BB width
        atr_score = min(s.atr_ratio / self._vol_high, 1.0)
        bb_score = min(s.bb_width / self._bb_wide, 1.0) if self._bb_wide > 0 else 0.5
        return 0.6 * atr_score + 0.4 * bb_score

    def _score_trend_strength(self, s: MarketState) -> float:
        """0 = no trend, 1 = strong trend."""
        adx_norm = min(s.adx_value / 50.0, 1.0)   # ADX 50+ = max trend
        r2_contrib = s.trend_r2                     # already 0-1
        hurst_contrib = max(0, (s.hurst - 0.5) * 4)  # 0 if Hurst<=0.5, 1 if Hurst>=0.75

        return 0.5 * adx_norm + 0.3 * r2_contrib + 0.2 * min(hurst_contrib, 1.0)

    def _score_momentum(self, s: MarketState) -> float:
        """−1 (bearish) to +1 (bullish)."""
        # RSI: 0→-1, 50→0, 100→+1
        rsi_score = (s.rsi - 50) / 50

        # EMA alignment: already -1 to +1
        ema_score = s.ema_alignment

        # ROC: normalize (typical range ±1%)
        roc_score = max(-1, min(1, s.roc / 0.5))

        return 0.3 * rsi_score + 0.4 * ema_score + 0.3 * roc_score

    def _score_volume(self, s: MarketState) -> float:
        """0 = very low volume, 1 = very high volume."""
        ratio_score = min(s.volume_ratio / 3.0, 1.0)  # 3x avg = max
        trend_score = min(s.volume_trend / 2.0, 1.0)   # 2x trend = max
        return 0.6 * ratio_score + 0.4 * trend_score

    def _score_structure(self, s: MarketState) -> float:
        """0 = indecisive candles, 1 = decisive/strong candles."""
        return s.body_ratio  # high body ratio = strong conviction

    # -------------------------------------------------------------------------
    # Regime determination
    # -------------------------------------------------------------------------
    def _determine_regime(
        self, s: MarketState, vol_score: float, trend_str: float
    ) -> Regime:
        """Determine raw regime from scores."""

        is_squeeze = (
            s.atr_ratio < self._vol_low
            and s.bb_width < self._bb_squeeze
            and s.vol_percentile < 0.2
        )
        if is_squeeze:
            return Regime.SQUEEZE

        is_breakout = (
            s.atr_ratio > self._vol_high
            and s.volume_ratio > 1.5
            and self._prev_regime in (Regime.SQUEEZE, Regime.RANGING)
        )
        if is_breakout:
            return Regime.BREAKOUT

        is_trending = (
            s.adx_value > self._adx_trending
            and s.hurst > self._hurst_trending
            and trend_str > 0.4
        )
        if is_trending:
            return Regime.TRENDING

        is_volatile = (
            vol_score > 0.7
            and trend_str < 0.3
        )
        if is_volatile:
            return Regime.VOLATILE

        return Regime.RANGING

    def _apply_persistence(self, raw_regime: Regime) -> Regime:
        """Require N bars of consistent signal before switching regime."""
        if raw_regime == self._prev_regime:
            self._regime_count += 1
            self._pending_regime = None
            self._pending_count = 0
            return self._prev_regime

        # New regime candidate
        if raw_regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_regime
            self._pending_count = 1

        # Confirm after persistence threshold
        if self._pending_count >= self._regime_persistence:
            self._prev_regime = raw_regime
            self._regime_count = self._pending_count
            self._pending_regime = None
            self._pending_count = 0
            return raw_regime

        return self._prev_regime

    def _determine_direction(self, s: MarketState, momentum: float) -> Direction:
        if momentum > 0.2:
            return Direction.BULLISH
        elif momentum < -0.2:
            return Direction.BEARISH
        return Direction.NEUTRAL

    def _compute_confidence(
        self, s: MarketState, regime: Regime, vol_score: float, trend_str: float
    ) -> float:
        """Confidence in the regime classification (0-1)."""
        base = 0.3

        if regime == Regime.TRENDING:
            base += 0.4 * trend_str + 0.2 * (s.adx_value / 50)
        elif regime == Regime.RANGING:
            ranging_evidence = max(0, 1 - trend_str * 2)
            base += 0.4 * ranging_evidence
        elif regime == Regime.BREAKOUT:
            base += 0.3 * vol_score + 0.2 * min(s.volume_ratio / 2, 1)
        elif regime == Regime.SQUEEZE:
            squeeze_evidence = max(0, 1 - vol_score * 2)
            base += 0.4 * squeeze_evidence
        elif regime == Regime.VOLATILE:
            base += 0.3 * vol_score

        # Persistence bonus: longer regime = more confidence
        persistence_bonus = min(self._regime_count / 20, 0.15)
        return min(base + persistence_bonus, 1.0)

    # -------------------------------------------------------------------------
    # Per-strategy activation scores
    # -------------------------------------------------------------------------
    def _trend_activation(
        self, s: MarketState, regime: Regime, trend_str: float, vol_score: float
    ) -> float:
        """How suitable is trend-following right now? (0-1)"""
        if regime == Regime.TRENDING:
            return 0.5 + 0.5 * trend_str
        elif regime == Regime.BREAKOUT:
            return 0.4 + 0.3 * vol_score
        return 0.1

    def _mr_activation(
        self, s: MarketState, regime: Regime, trend_str: float, vol_score: float
    ) -> float:
        """How suitable is mean-reversion right now? (0-1)"""
        if regime == Regime.RANGING:
            return 0.6 + 0.3 * (1 - trend_str) + 0.1 * (1 - vol_score)
        elif regime == Regime.VOLATILE:
            return 0.3  # possible but risky
        return 0.05

    def _breakout_activation(
        self, s: MarketState, regime: Regime, vol_score: float, volume_score: float
    ) -> float:
        """How suitable is breakout trading right now? (0-1)"""
        if regime == Regime.BREAKOUT:
            return 0.6 + 0.2 * vol_score + 0.2 * volume_score
        elif regime == Regime.SQUEEZE:
            return 0.3  # prepare for breakout
        return 0.05

    def _scalp_activation(
        self, s: MarketState, regime: Regime, vol_score: float, volume_score: float
    ) -> float:
        """How suitable is scalping right now? (0-1)"""
        if regime == Regime.VOLATILE:
            return 0.5 + 0.3 * vol_score + 0.2 * volume_score
        elif regime == Regime.RANGING and volume_score > 0.5:
            return 0.4
        return 0.1

    def _fade_activation(
        self, s: MarketState, regime: Regime, vol_score: float, momentum: float
    ) -> float:
        """How suitable is fading extremes right now? (0-1)"""
        # Fade works best after extreme momentum in volatile/ranging
        extreme_momentum = abs(momentum) > 0.7
        if extreme_momentum and regime in (Regime.VOLATILE, Regime.RANGING):
            return 0.5 + 0.3 * vol_score
        return 0.05
