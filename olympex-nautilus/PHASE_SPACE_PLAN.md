# Phase Space Plan: Thị trường như Hệ Vật Lý

## Vấn đề hiện tại

1. **Parameter-dependent**: Thay đổi EMA(9,21) → EMA(14,50) = kết quả hoàn toàn khác
2. **Discrete labels**: 5 regime rời rạc (trending/ranging/breakout/squeeze/volatile) — thực tế thị trường là phổ liên tục
3. **Indicator-based**: Đo indicator chứ không đo thị trường

## Thiết kế mới: Phase Space Engine

### Tầng 1: Statistical Properties (Không phụ thuộc tham số)

Đo **thuộc tính nội tại** của phân phối returns:

| Metric | Ý nghĩa | Cách tính |
|--------|---------|-----------|
| `realized_vol` | Biến động thực | std(returns) — không cần ATR period |
| `skewness` | Lệch phân phối | Moment bậc 3 — thị trường thiên hướng lên/xuống |
| `kurtosis` | Đuôi béo | Moment bậc 4 — xác suất extreme moves |
| `autocorrelation` | Tự tương quan | returns[t] vs returns[t-1] — trend hay mean-revert |
| `hurst` | Persistence | Rescaled range — bản chất trending/ranging |
| `entropy` | Hỗn loạn | Shannon entropy của returns distribution |

**Tại sao tốt hơn?** Đổi window size → giá trị thay đổi nhẹ, nhưng RANKING (high/low) giữ nguyên. Variance cao vẫn là variance cao dù đo trên 20 hay 50 bars.

### Tầng 2: Multi-Scale Analysis (Robust)

Tính MỖI metric trên NHIỀU scale rồi tổng hợp:

```
scales = [5, 10, 20, 50, 100]  # bars

Ví dụ volatility:
  vol_5  = std(returns[-5:])
  vol_10 = std(returns[-10:])
  vol_20 = std(returns[-20:])
  vol_50 = std(returns[-50:])
  vol_100 = std(returns[-100:])

  → volatility_score = weighted_percentile_rank(vol_multi_scale)
  → volatility_trend = vol_short / vol_long  (tăng hay giảm?)
```

**Tại sao tốt hơn?** Không phụ thuộc vào BẤT KỲ period nào. Nếu volatility cao trên cả 5 scale → chắc chắn cao. Nếu chỉ cao trên scale ngắn → mới bắt đầu tăng.

### Tầng 3: Distribution-Based (Phase Detection)

Thay vì label "trending" hay "ranging", dùng **trạng thái phân phối**:

- **Regime A** (trending): returns có mean ≠ 0, autocorrelation > 0, Hurst > 0.5
- **Regime B** (ranging): returns có mean ≈ 0, autocorrelation < 0, Hurst < 0.5
- **Regime C** (volatile): returns có kurtosis cao, variance cao, entropy cao

Nhưng thay vì chọn A/B/C, output là **vector xác suất liên tục**: `[0.6, 0.3, 0.1]`

## Output: PhaseState

```python
@dataclass
class PhaseState:
    """Trạng thái trong không gian pha liên tục."""

    # === POSITION (Vị trí hiện tại trên phổ liên tục) ===
    # Mỗi trục là 0-1, KHÔNG phải label
    volatility: float       # 0=rất yên tĩnh, 1=rất biến động
    trend: float            # -1=downtrend mạnh, 0=sideway, +1=uptrend mạnh
    mean_reversion: float   # 0=không có, 1=mean-reversion rõ ràng
    momentum: float         # -1=bearish cực, 0=trung tính, +1=bullish cực
    liquidity: float        # 0=thanh khoản thấp, 1=thanh khoản cao

    # === VELOCITY (Tốc độ chuyển pha) ===
    d_volatility: float     # volatility đang tăng (+) hay giảm (-)
    d_trend: float          # trend đang mạnh lên (+) hay yếu đi (-)
    d_mean_reversion: float
    d_momentum: float
    d_liquidity: float

    # === ACCELERATION (Gia tốc chuyển pha) ===
    dd_volatility: float    # tốc độ thay đổi volatility đang tăng hay giảm
    dd_trend: float
    dd_momentum: float

    # === FORCE & INERTIA ===
    inertia: float          # 0-1, trạng thái hiện tại ổn định đến mức nào
    net_force: float        # -1 to +1, lực ròng đang đẩy thị trường
    force_alignment: float  # Các lực cùng hướng (1) hay trái hướng (0)?

    # === ANOMALY ===
    whale_score: float      # 0=bình thường, 1=hoạt động bất thường
    regime_break: float     # 0=ổn định, 1=đang phá vỡ regime

    # === STRATEGY SCORES (derived) ===
    trend_suitability: float
    mr_suitability: float
    breakout_suitability: float
    scalp_suitability: float
    fade_suitability: float

    # Meta
    confidence: float       # Tổng confidence (dựa trên data sufficiency)
    bars_processed: int
```

## Cách tính cụ thể

### Position: Multi-scale Statistical

```python
def compute_volatility(returns, volumes):
    """Volatility score 0-1 dựa trên statistical properties."""
    scores = []
    for scale in [5, 10, 20, 50, 100]:
        if len(returns) >= scale:
            window = returns[-scale:]
            scores.append(np.std(window))

    # Percentile rank trong lịch sử
    current = np.mean(scores)  # average across scales
    return percentile_rank(current, vol_history)
```

### Velocity: Finite Difference

```python
def compute_velocity(position_history):
    """d(position)/dt — rate of phase change."""
    if len(position_history) < 2:
        return 0.0
    return position_history[-1] - position_history[-2]
```

### Acceleration

```python
def compute_acceleration(velocity_history):
    """d²(position)/dt² — is transition speeding up?"""
    if len(velocity_history) < 2:
        return 0.0
    return velocity_history[-1] - velocity_history[-2]
```

### Inertia (Resistance to change)

```python
def compute_inertia(position_history, window=20):
    """How stable is the current state? Low variance = high inertia."""
    if len(position_history) < window:
        return 0.5
    recent = position_history[-window:]
    stability = 1.0 - min(np.std(recent) * 10, 1.0)
    return stability
```

### Whale Detection

```python
def compute_whale_score(volume, returns, vol_history, return_history):
    """Detect anomalous activity."""
    vol_zscore = (volume - np.mean(vol_history)) / np.std(vol_history)
    ret_zscore = abs(returns[-1]) / np.std(return_history)

    # Z-score > 3 = anomaly
    whale = max(0, min(1, (max(vol_zscore, ret_zscore) - 2) / 3))
    return whale
```

### Strategy Suitability (Derived from phase position)

```python
def compute_strategy_scores(phase: PhaseState):
    """Strategy scores derived from continuous phase, not labels."""

    # Trend following: high trend + high momentum alignment + not mean-reverting
    trend_suit = (
        abs(phase.trend) * 0.4 +
        abs(phase.momentum) * 0.3 +
        (1 - phase.mean_reversion) * 0.2 +
        phase.liquidity * 0.1
    ) * (1 if phase.trend * phase.momentum > 0 else 0.3)  # alignment bonus

    # Mean reversion: high mean_reversion + low trend + moderate volatility
    mr_suit = (
        phase.mean_reversion * 0.5 +
        (1 - abs(phase.trend)) * 0.3 +
        (0.3 < phase.volatility < 0.7) * 0.2
    )

    # Breakout: volatility increasing + high momentum + high volume
    breakout_suit = (
        max(0, phase.d_volatility) * 0.3 +
        abs(phase.momentum) * 0.3 +
        phase.liquidity * 0.2 +
        phase.regime_break * 0.2
    )

    # Scalp: high volatility + no clear trend + good liquidity
    scalp_suit = (
        phase.volatility * 0.4 +
        (1 - abs(phase.trend)) * 0.3 +
        phase.liquidity * 0.3
    )

    # Fade: extreme momentum + high volatility + mean-reversion tendency
    fade_suit = (
        (abs(phase.momentum) > 0.7) * 0.4 +
        phase.volatility * 0.3 +
        phase.mean_reversion * 0.3
    )
```

## File structure

```
strategies/
├── phase_space.py          # NEW: PhaseState + PhaseSpaceEngine
├── market_measure.py       # KEEP (backward compat, but deprecated)
├── regime_classifier.py    # KEEP (backward compat, but deprecated)
├── olympex_strategy.py     # UPDATE: use PhaseSpaceEngine
├── strategy_modules.py     # UPDATE: accept PhaseState instead of RegimeSignal
└── __init__.py
```

## Implementation Steps

1. Create `phase_space.py` with PhaseState and PhaseSpaceEngine
2. PhaseSpaceEngine computes all statistical properties + multi-scale + derivatives
3. Update `olympex_strategy.py` to use PhaseSpaceEngine alongside (not replacing) existing system
4. Update strategy modules to accept PhaseState for signal gating
5. Keep old system for A/B comparison in backtest

## Câu hỏi mở

- Whale detection chỉ PHÁT HIỆN được, không DỰ ĐOÁN được. Đây là giới hạn của bất kỳ hệ thống nào.
- "Lực cần thiết để chuyển pha" = inertia. Đo bằng stability của phase position qua thời gian.
- Phase transition speed = velocity. Nếu velocity cao + acceleration dương → thị trường đang chuyển pha nhanh.
