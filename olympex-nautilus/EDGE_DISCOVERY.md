# Edge Discovery — Tìm Vùng Xác Suất Lệch

## Bài Toán

```
CHO:  State vector S(t) ∈ R³³ tại mỗi thời điểm t
      Return thực tế r(t+1) xảy ra SAU đó

TÌM:  Những vùng trong không gian state mà:
      P(r(t+1) > 0 | S(t) ∈ vùng) ≠ 0.5  (có ý nghĩa thống kê)

CHỨNG MINH:  Thiên hướng đó không phải ngẫu nhiên VÀ tồn tại ngoài mẫu
```

**Nói đơn giản**: Ở toạ độ nào trên bản đồ 33 chiều, giá có xu hướng đi 1 hướng
nhiều hơn hướng kia?

---

## Hai Hướng Tiếp Cận — A/B Test

### Approach A: Bottom-Up (Data-Driven)

> "Để data tự nói. Quét không gian state, tìm vùng nào return lệch."

**Không có giả thuyết trước** — thuần tuý thống kê.

### Approach B: Top-Down (Theory-Driven)

> "Dựa vào regime + transition logic để suy luận khi nào giá có hướng."

**Có giả thuyết** — dùng hiểu biết về 4 regime và chuyển pha.

### A/B Test Framework

```
Cùng data → cùng measurement engine → 2 signal generator khác nhau → so sánh

                    ┌─────────────────┐
                    │  OHLCV Data     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Measurement    │
                    │  Engine (33-D)  │
                    └───┬─────────┬───┘
                        │         │
               ┌────────▼───┐ ┌───▼────────┐
               │ Approach A │ │ Approach B  │
               │ Data-Driven│ │ Theory-Drv  │
               └────────┬───┘ └───┬────────┘
                        │         │
               ┌────────▼───┐ ┌───▼────────┐
               │ Signals A  │ │ Signals B   │
               └────────┬───┘ └───┬────────┘
                        │         │
               ┌────────▼─────────▼────────┐
               │  Backtest Engine          │
               │  (cùng data, cùng rules)  │
               └───────────────────────────┘
                        │
               So sánh: hit rate, PnL, Sharpe, max DD
```

---

## Approach A: Bottom-Up (Data-Driven)

### Bước A1: Tìm chiều có sức dự báo hướng

33 chiều quá nhiều. Bước đầu: tìm **chiều nào liên quan đến hướng giá**.

```python
def scan_predictive_dimensions(state_history, returns):
    """
    Với mỗi chiều trong 33 chiều:
    - Chia thành 5 quintile bins
    - Tính hit_rate (% return dương) trong mỗi bin
    - Nếu hit_rate thay đổi đơn điệu → chiều đó có sức dự báo
    """
    results = {}

    for dim_name in ALL_33_DIMENSIONS:
        dim_values = [s[dim_name] for s in state_history]

        # Chia thành 5 bin (quintile)
        quintiles = np.percentile(dim_values, [20, 40, 60, 80])
        bins = np.digitize(dim_values, quintiles)  # 0,1,2,3,4

        hit_rates = []
        avg_returns = []
        counts = []

        for bin_id in range(5):
            mask = (bins == bin_id)
            if mask.sum() < 30:  # minimum sample
                continue
            bin_returns = returns[mask]
            hit_rates.append((bin_returns > 0).mean())
            avg_returns.append(bin_returns.mean())
            counts.append(mask.sum())

        # Đo tính đơn điệu (monotonicity)
        if len(hit_rates) >= 4:
            # Spearman rank correlation giữa bin_id và hit_rate
            monotonicity = spearman_corr(range(len(hit_rates)), hit_rates)

            # Statistical significance
            spread = max(hit_rates) - min(hit_rates)

            results[dim_name] = {
                'hit_rates': hit_rates,
                'avg_returns': avg_returns,
                'monotonicity': monotonicity,    # -1 to +1
                'spread': spread,                 # max - min hit rate
                'counts': counts,
                'predictive': abs(monotonicity) > 0.7 and spread > 0.06
            }

    # Xếp hạng theo |monotonicity| × spread
    ranked = sorted(results.items(),
                    key=lambda x: abs(x[1]['monotonicity']) * x[1]['spread'],
                    reverse=True)
    return ranked
```

**Output ví dụ:**

```
Rank  Dimension           Monotonicity  Spread   Hit rates by quintile
─────────────────────────────────────────────────────────────────────
  1   skewness            +0.95         0.12     [44%, 47%, 50%, 53%, 56%]
  2   autocorrelation     +0.88         0.10     [45%, 47%, 50%, 52%, 55%]
  3   Δautocorr           +0.82         0.09     [45%, 48%, 50%, 52%, 54%]
  4   pv_corr             +0.75         0.08     [46%, 48%, 50%, 52%, 54%]
  ...
 30   corr(Δvol,Δskew)    +0.12         0.02     [49%, 50%, 50%, 51%, 51%]  ← vô dụng
```

→ Giữ lại **top 5-8 chiều** có predictive power.

### Bước A2: Tìm tổ hợp điều kiện (combo)

Với top N chiều, tìm **vùng giao nhau** có edge mạnh hơn:

```python
def find_edge_combos(state_history, returns, top_dims, max_combo=3):
    """
    Grid search trên tổ hợp 2-3 chiều mạnh nhất.
    Tìm vùng có hit_rate cao nhất VÀ đủ sample.
    """
    edges = []

    for combo in combinations(top_dims, max_combo):
        # Với mỗi combo, chia mỗi chiều thành HIGH/LOW (trên/dưới median)
        for bin_config in product(['LOW', 'HIGH'], repeat=len(combo)):
            mask = np.ones(len(returns), dtype=bool)

            for dim, bin_val in zip(combo, bin_config):
                median = np.median([s[dim] for s in state_history])
                dim_values = np.array([s[dim] for s in state_history])
                if bin_val == 'HIGH':
                    mask &= (dim_values > median)
                else:
                    mask &= (dim_values <= median)

            count = mask.sum()
            if count < 30:  # minimum sample size
                continue

            bin_returns = returns[mask]
            hit_rate = (bin_returns > 0).mean()
            avg_ret = bin_returns.mean()

            # Binomial test: hit_rate có khác 50% thật không?
            p_value = binomial_test(hit_rate, count, 0.5)

            if p_value < 0.05:
                edges.append({
                    'dims': combo,
                    'bins': bin_config,
                    'hit_rate': hit_rate,
                    'avg_return': avg_ret,
                    'count': count,
                    'p_value': p_value,
                    'direction': 'LONG' if hit_rate > 0.5 else 'SHORT'
                })

    return sorted(edges, key=lambda x: abs(x['hit_rate'] - 0.5), reverse=True)
```

**Output ví dụ:**

```
Edge #1: skewness=HIGH AND autocorr=HIGH AND Δautocorr=HIGH
         → hit_rate=59.2%, avg_return=+0.03%, count=847, p<0.001
         → Direction: LONG

Edge #2: skewness=LOW AND autocorr=LOW AND vol_anomaly=HIGH
         → hit_rate=57.1%, avg_return=-0.025%, count=523, p=0.003
         → Direction: SHORT

Edge #3: pv_corr=HIGH AND Δvol=HIGH AND autocorr=HIGH
         → hit_rate=56.8%, avg_return=+0.028%, count=412, p=0.008
         → Direction: LONG
```

### Bước A3: Out-of-sample validation

```python
def validate_edges(data, split_ratio=0.67):
    """
    In-sample: 67% đầu → TÌM edges
    Out-of-sample: 33% sau → KIỂM TRA edges
    """
    split = int(len(data) * split_ratio)
    in_sample = data[:split]
    out_sample = data[split:]

    # Tìm edges trên in-sample
    edges = find_edge_combos(in_sample.states, in_sample.returns, top_dims)

    # Kiểm tra trên out-of-sample
    validated = []
    for edge in edges:
        oos_mask = apply_edge_condition(out_sample.states, edge)
        oos_returns = out_sample.returns[oos_mask]

        if len(oos_returns) < 15:
            continue

        oos_hit_rate = (oos_returns > 0).mean()
        oos_avg_ret = oos_returns.mean()

        # Edge vẫn hoạt động ngoài mẫu?
        degradation = abs(edge['hit_rate'] - oos_hit_rate)

        validated.append({
            **edge,
            'oos_hit_rate': oos_hit_rate,
            'oos_avg_return': oos_avg_ret,
            'oos_count': len(oos_returns),
            'degradation': degradation,
            'still_valid': (
                (edge['direction'] == 'LONG' and oos_hit_rate > 0.52) or
                (edge['direction'] == 'SHORT' and oos_hit_rate < 0.48)
            )
        })

    return validated
```

**Acceptance criteria cho Approach A:**

| Tiêu chí | Ngưỡng |
|-----------|--------|
| In-sample hit rate | > 55% hoặc < 45% |
| In-sample count | >= 30 |
| In-sample p-value | < 0.05 |
| Out-of-sample hit rate cùng hướng | > 52% hoặc < 48% |
| Degradation | < 5% (hit rate giảm không quá 5 điểm %) |

---

## Approach B: Top-Down (Theory-Driven)

### Giả thuyết: Regime + Transition → Hướng giá

Dựa vào hiểu biết về 4 regime và physics của chuyển pha:

#### Giả thuyết B1: MOMENTUM regime → giá tiếp tục theo hướng hiện tại

```python
def signal_momentum_continuation(state, membership):
    """
    Khi: MOMENTUM dominant (>60%) VÀ autocorr dương VÀ volume confirm
    Thì: Giá sẽ tiếp tục theo hướng skewness
    """
    if membership['MOMENTUM'] < 0.6:
        return None
    if state.autocorrelation < 0.1:
        return None
    if state.pv_corr < 0.2:
        return None

    direction = 'LONG' if state.skewness > 0 else 'SHORT'
    strength = membership['MOMENTUM'] * abs(state.autocorrelation) * state.pv_corr

    return Signal(direction=direction, strength=strength, reason='momentum_continuation')
```

#### Giả thuyết B2: MEAN-REVERT regime → giá đảo ngược từ extreme

```python
def signal_mean_reversion(state, membership, price, mean_price, std_price):
    """
    Khi: MEAN-REVERT dominant (>60%) VÀ autocorr âm VÀ giá ở extreme
    Thì: Giá sẽ quay về mean
    """
    if membership['REVERT'] < 0.6:
        return None
    if state.autocorrelation > -0.1:
        return None

    zscore = (price - mean_price) / std_price

    if zscore > 1.5:
        direction = 'SHORT'  # fade overbought
    elif zscore < -1.5:
        direction = 'LONG'   # fade oversold
    else:
        return None  # không ở extreme

    strength = membership['REVERT'] * abs(state.autocorrelation) * abs(zscore) / 3

    return Signal(direction=direction, strength=strength, reason='mean_reversion')
```

#### Giả thuyết B3: Chuyển pha QUIET→MOMENTUM → breakout theo hướng volume

```python
def signal_regime_transition(state, membership, transition_scores):
    """
    Khi: transition score QUIET→MOMENTUM > 60
    VÀ  velocity hướng về MOMENTUM
    Thì: Follow breakout direction
    """
    qt_score = transition_scores.get(('QUIET', 'MOMENTUM'), 0)

    if qt_score < 60:
        return None

    # Hướng breakout = hướng của skewness + autocorr mới xuất hiện
    if state.skewness > 0 and state.autocorrelation > 0:
        direction = 'LONG'
    elif state.skewness < 0 and state.autocorrelation > 0:
        direction = 'SHORT'
    else:
        return None

    strength = qt_score / 100 * abs(state.skewness)

    return Signal(direction=direction, strength=strength, reason='quiet_to_momentum')
```

#### Giả thuyết B4: CHAOS → Không trade (survival)

```python
def signal_chaos_filter(membership):
    """
    Khi: CHAOS membership > 40%
    Thì: Giảm size hoặc đóng hết
    """
    if membership['CHAOS'] > 0.7:
        return Signal(direction='FLAT', strength=1.0, reason='chaos_close_all')
    elif membership['CHAOS'] > 0.4:
        return Signal(direction='REDUCE', strength=membership['CHAOS'], reason='chaos_reduce')
    return None
```

#### Giả thuyết B5: Dynamics divergence → Chuẩn bị đảo chiều

```python
def signal_divergence(state):
    """
    Khi: Volume tăng (Δvol_anomaly > 0) NHƯNG momentum giảm (Δautocorr < 0)
    Thì: Regime hiện tại sắp hết, chuẩn bị đảo
    """
    vol_increasing = state.d_vol_anomaly > 0.1
    momentum_decreasing = state.d_autocorrelation < -0.05

    if vol_increasing and momentum_decreasing:
        # Divergence: volume nói "có chuyện" nhưng momentum nói "hết sức"
        return Signal(direction='CLOSE_TREND', strength=0.7, reason='vol_momentum_divergence')
    return None
```

### Bước B_Test: Kiểm tra từng giả thuyết

Mỗi giả thuyết được test **riêng biệt** trên dữ liệu lịch sử:

```python
def test_hypothesis(signal_func, state_history, returns):
    """
    Chạy signal function trên toàn bộ history.
    Đếm: bao nhiêu lần signal đúng, bao nhiêu lần sai.
    """
    signals = []
    for t, state in enumerate(state_history[:-1]):
        signal = signal_func(state, ...)
        if signal is not None:
            actual_return = returns[t+1]
            correct = (
                (signal.direction == 'LONG' and actual_return > 0) or
                (signal.direction == 'SHORT' and actual_return < 0)
            )
            signals.append({
                'time': t,
                'direction': signal.direction,
                'strength': signal.strength,
                'actual_return': actual_return,
                'correct': correct
            })

    hit_rate = sum(s['correct'] for s in signals) / len(signals) if signals else 0
    avg_return = np.mean([
        s['actual_return'] if s['direction'] == 'LONG' else -s['actual_return']
        for s in signals
    ]) if signals else 0

    return {
        'hypothesis': signal_func.__name__,
        'total_signals': len(signals),
        'hit_rate': hit_rate,
        'avg_return': avg_return,
        'p_value': binomial_test(hit_rate, len(signals), 0.5)
    }
```

**Acceptance criteria cho Approach B:**

| Tiêu chí | Ngưỡng |
|-----------|--------|
| Signals per 1000 bars | >= 20 (đủ thường xuyên) |
| Hit rate | > 53% (khiêm tốn nhưng consistent) |
| p-value | < 0.05 |
| Out-of-sample hold | Cùng hướng, hit rate > 51% |

---

## A/B Test: So sánh 2 Approach

### Cùng điều kiện

| Yếu tố | Giống nhau |
|---------|------------|
| Data | Cùng BTC 1-min candles |
| Measurement | Cùng 33-dim state vector |
| Position sizing | Cùng quy tắc (% equity) |
| Stop loss / Take profit | Cùng ATR-based |
| Slippage / Fees | Cùng (taker 0.05%) |
| Backtest period | Cùng train/test split |

### Metrics so sánh

| Metric | Approach A (Data-Driven) | Approach B (Theory-Driven) |
|--------|--------------------------|----------------------------|
| Total signals | ? | ? |
| Hit rate | ? | ? |
| Average return per trade | ? | ? |
| Profit factor | ? | ? |
| Max drawdown | ? | ? |
| Sharpe ratio | ? | ? |
| Signal stability (ít flip) | ? | ? |
| Out-of-sample degradation | ? | ? |
| Overfitting risk | CAO (vì data-mined) | THẤP (vì theory-based) |
| Adaptability | CAO (tự tìm edge mới) | THẤP (cần update giả thuyết) |

### Kỳ vọng

- **Approach A** sẽ có hit rate IN-SAMPLE cao hơn (vì tối ưu hoá trên data)
  nhưng OUT-OF-SAMPLE có thể degrade nhiều hơn (overfitting risk)
- **Approach B** sẽ có hit rate IN-SAMPLE thấp hơn
  nhưng OUT-OF-SAMPLE stable hơn (vì dựa trên logic, không curve-fit)
- **Approach C** (kết hợp): Dùng B làm bộ lọc, A làm tinh chỉnh → có thể best of both

---

## Approach C: Kết Hợp — B Lọc, A Tinh Chỉnh

```
Bước 1: Approach B xác định "khi nào có thể trade" (regime + transition filter)
Bước 2: Trong những lúc B cho phép, Approach A tìm "vùng nào có edge mạnh nhất"
Bước 3: Trade chỉ khi CẢ A VÀ B đồng ý

→ Ít signals hơn, nhưng mỗi signal chất lượng hơn
```

```python
def combined_signal(state, membership, transition_scores, edge_conditions):
    # B: Có nên trade không? (theory filter)
    b_signal = approach_b_signal(state, membership, transition_scores)
    if b_signal is None or b_signal.direction in ['FLAT', 'REDUCE']:
        return b_signal  # Respect CHAOS filter

    # A: Có edge statistical không? (data confirmation)
    a_signal = approach_a_signal(state, edge_conditions)
    if a_signal is None:
        return None  # B cho phép nhưng A không tìm thấy edge

    # Cả hai đồng ý?
    if a_signal.direction == b_signal.direction:
        # Agreement → signal mạnh
        combined_strength = (a_signal.strength + b_signal.strength) / 2
        return Signal(
            direction=a_signal.direction,
            strength=combined_strength,
            reason=f'{b_signal.reason}+{a_signal.reason}'
        )
    else:
        # Conflict → không trade
        return None
```

---

## Files Cần Tạo

| File | Mục đích |
|------|----------|
| `strategies/edge_scanner.py` | Approach A: scan dimensions, find combos, validate |
| `strategies/theory_signals.py` | Approach B: 5 giả thuyết → signals |
| `strategies/signal_combiner.py` | Approach C: kết hợp A+B |
| `tests/test_edge_scanner.py` | Test Approach A trên synthetic + real data |
| `tests/test_theory_signals.py` | Test từng giả thuyết B riêng biệt |
| `tests/test_ab_comparison.py` | A/B test framework: cùng data, so sánh metrics |
| `scripts/run_edge_discovery.py` | Script chạy Approach A offline: scan → report |
| `scripts/run_ab_test.py` | Script chạy A/B test → output comparison table |

---

## Thứ Tự Thực Hiện

### Prerequisite: Measurement Engine đã pass tất cả test (xem MEASUREMENT_ENGINE.md)

### Phase 1: Approach A
1. Implement `edge_scanner.py`
2. Chạy `run_edge_discovery.py` trên BTC data → xuất báo cáo chiều nào có edge
3. Test out-of-sample → xác nhận hoặc loại bỏ

### Phase 2: Approach B
4. Implement `theory_signals.py` với 5 giả thuyết
5. Test từng giả thuyết riêng biệt → giữ cái pass, loại cái fail
6. Test tổng hợp tất cả signals

### Phase 3: A/B Test
7. Implement `run_ab_test.py` → chạy cả A và B trên cùng data
8. So sánh tất cả metrics
9. Nếu C (kết hợp) tốt hơn → implement `signal_combiner.py`

### Phase 4: Backtest
10. Tích hợp signal generator thắng cuộc vào `olympex_strategy.py`
11. Chạy full backtest trên NautilusTrader
12. So sánh với strategy cũ (indicator-based)
