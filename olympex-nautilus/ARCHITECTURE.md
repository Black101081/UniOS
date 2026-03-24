# Olympex Trading System — Architecture

## Core Philosophy
> "Không có chiến lược nào phù hợp mọi thị trường. Đo thị trường trước, chọn chiến lược sau."

## System Layers

```
┌─────────────────────────────────────────────────────┐
│                  RISK OVERLAY                        │
│  Position sizing, drawdown limits, fat-tail guard    │
├─────────────────────────────────────────────────────┤
│              STRATEGY ROUTER                         │
│  Regime → Strategy mapping, confidence gating        │
├──────────┬──────────┬──────────┬────────┬───────────┤
│  TREND   │   MR     │ BREAKOUT │ SCALP  │   FADE    │
│ follower │ revert   │ momentum │ quick  │ extreme   │
├──────────┴──────────┴──────────┴────────┴───────────┤
│           REGIME CLASSIFIER                          │
│  Multi-regime: trending, ranging, breakout,          │
│  squeeze, volatile — with confidence scores          │
├─────────────────────────────────────────────────────┤
│         MARKET MEASUREMENT ENGINE                    │
│  Volatility │ Trend │ Momentum │ Volume │ Structure  │
│  (macro)    │       │          │        │ (micro)    │
├─────────────────────────────────────────────────────┤
│              RAW MARKET DATA                         │
│  1-min OHLCV bars from Hyperliquid                   │
└─────────────────────────────────────────────────────┘
```

## Layer 1: Market Measurement Engine

Continuously measures 5 dimensions on every bar:

### Volatility Dimension
- `atr_ratio`: Current ATR / SMA(ATR, 100) — relative volatility
- `bb_width`: (BB_upper - BB_lower) / BB_middle — normalized bandwidth
- `vol_percentile`: Where current realized vol sits in 200-bar history

### Trend Dimension
- `adx_value`: Average Directional Index — trend strength (0-100)
- `ema_alignment`: Price vs EMA_fast vs EMA_slow alignment score
- `linear_r2`: R² of linear regression on last N closes — trend quality
- `hurst`: Rolling Hurst exponent — persistence vs mean reversion

### Momentum Dimension
- `rsi`: Relative Strength Index
- `macd_histogram`: MACD - Signal line
- `roc`: Rate of Change (close vs N bars ago)
- `momentum_divergence`: Price making new high but RSI not (bearish div)

### Volume Dimension
- `volume_ratio`: Current vol / SMA(vol, 20) — relative volume
- `volume_trend`: SMA(vol, 5) / SMA(vol, 20) — volume momentum
- `buy_sell_pressure`: Estimated from candle body position in range

### Microstructure Dimension
- `candle_body_ratio`: |close-open| / (high-low) — conviction
- `wick_ratio`: Upper/lower wick relative to body — rejection
- `gap_size`: Current bar open vs previous close

## Layer 2: Regime Classifier

Takes all measurements and classifies market into regimes:

| Regime | Key Signals | Confidence Factors |
|--------|------------|-------------------|
| **TRENDING_BULL** | ADX>25, EMA aligned up, Hurst>0.55 | ADX level, R² |
| **TRENDING_BEAR** | ADX>25, EMA aligned down, Hurst>0.55 | ADX level, R² |
| **RANGING** | ADX<20, low BB width, Hurst<0.45 | How flat ADX is |
| **BREAKOUT** | Vol expanding, price at range extreme | BB squeeze then expand |
| **SQUEEZE** | BB width contracting, very low ATR | Width percentile |
| **VOLATILE** | High ATR, no trend, big wicks | ATR ratio, body ratio |

### Dynamic Weighting
Weights of each dimension change per regime:
- In **trending**: Trend dimension weight ↑, Volume confirms
- In **ranging**: Volatility dimension weight ↑, Momentum for mean-reversion timing
- In **breakout**: Volume dimension weight ↑↑, Volatility expansion key
- In **volatile**: Microstructure weight ↑, need candle confirmation

## Layer 3: Strategy Router

Maps regime + confidence → active strategy:

| Regime | Primary Strategy | Fallback | Confidence Gate |
|--------|-----------------|----------|-----------------|
| TRENDING_BULL | TrendFollower(LONG) | — | >0.3 |
| TRENDING_BEAR | TrendFollower(SHORT) | — | >0.3 |
| RANGING | MeanReversion | Fade | >0.4 |
| BREAKOUT | BreakoutMomentum | TrendFollower | >0.5 |
| SQUEEZE | Wait (no trade) | Breakout watch | — |
| VOLATILE | Scalper | Fade | >0.6 |

### Transition Rules
- **Regime change**: Close existing positions from old regime strategy
- **Low confidence**: Reduce position size by 50%
- **Conflicting signals**: Sit out (no new trades)

## Layer 4: Risk Overlay

Applies regardless of strategy:
- **Position sizing**: Based on ATR and regime confidence
- **Max drawdown**: Daily/session drawdown limit
- **Fat-tail guard**: If 3-sigma move detected, flatten everything
- **Correlation**: Don't double down in same direction across strategies

## Strategy Specifications

### TrendFollower
- Entry: EMA cross confirmed by ADX > threshold
- Exit: Trailing stop (ATR-based), regime flip
- Best in: TRENDING regimes with ADX > 25

### MeanReversion
- Entry: Price at BB extreme + RSI divergence
- Exit: Return to mean (BB middle) or time stop
- Best in: RANGING regimes with low ADX

### BreakoutMomentum
- Entry: Price breaks N-bar high/low with volume surge
- Exit: Momentum exhaustion (RSI extreme, volume fade)
- Best in: BREAKOUT regime (BB squeeze → expand)

### Scalper
- Entry: Quick mean-reversion on wicks in high-volatility
- Exit: Fixed pip target, very tight stop
- Best in: VOLATILE regime, high liquidity periods

### Fade
- Entry: Fade extreme moves (3-sigma candles)
- Exit: Partial reversion, tight stop
- Best in: VOLATILE/RANGING, after exhaustion candles
