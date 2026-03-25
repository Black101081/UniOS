# Phase Space Plan: Roadmap Tổng Thể

## Tổng quan hệ thống mới

```
OHLCV Data
    │
    ▼
┌───────────────────┐
│ MEASUREMENT ENGINE │  ← Phase 1 (MEASUREMENT_ENGINE.md)
│ 6 metrics gốc     │
│ 33-dim state vector│
│ 4 regime membership│
│ Transition scoring │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ EDGE DISCOVERY    │  ← Phase 2 (EDGE_DISCOVERY.md)
│ Approach A: Data  │
│ Approach B: Theory│
│ A/B Test          │
│ → Approach C: Mix │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ STRATEGY ENGINE   │  ← Phase 3 (future)
│ Signal generation │
│ Position sizing   │
│ Risk management   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ BACKTEST / LIVE   │  ← Phase 4 (future)
│ NautilusTrader    │
│ A/B comparison    │
└───────────────────┘
```

---

## Vấn đề với hệ thống CŨ

| Vấn đề | Ví dụ | Ảnh hưởng |
|---------|-------|-----------|
| Parameter-dependent | EMA(9,21) vs EMA(14,50) = kết quả khác | Curve-fitting, không generalise |
| Discrete labels | 5 regime rời rạc (trending/ranging/...) | Thị trường là phổ liên tục |
| Indicator-based | RSI, MACD, ADX = biến đổi giá với magic number | Đo indicator, không đo thị trường |
| Không có dynamics | Chỉ biết "đang ở đâu", không biết "đang đi đâu" | Bỏ lỡ chuyển pha |
| Không có edge validation | Trade dựa trên indicator cross = không rõ có edge thật không | Không biết lý do thắng/thua |

## Thiết kế mới: Từ gốc đến ngọn

### Phase 1: Measurement Engine

**Tài liệu**: `MEASUREMENT_ENGINE.md`

**Nguyên lý**: Chỉ dùng thuộc tính thống kê nội tại — những thứ thật sự đo được.

| Component | Input | Output |
|-----------|-------|--------|
| 6 Metric Gốc | OHLCV | vol, autocorr, skew, kurtosis, vol_anomaly, pv_corr |
| Multi-Scale | Mỗi metric × 5 windows | Consensus score (robust) |
| State Vector | 6 metrics × history | 33 chiều (vị trí + vận tốc + gia tốc + tương quan) |
| Regime Membership | 6 metrics | 4 regime weights (tổng = 1) |
| Transition Score | Membership history | Score 0-100 cho mỗi chuyển pha |

**Files**:
```
strategies/
├── stat_metrics.py        # 6 metric gốc + multi-scale
├── state_vector.py        # 33-dim: position + velocity + accel + cross-corr
└── regime_membership.py   # 4 regime weights + transition scoring
```

**Tests**:
```
tests/
├── test_metrics.py        # Unit test từng metric
├── test_robustness.py     # Window stability (ranking nhất quán >= 85%)
├── test_regime_synthetic.py # Classify đúng trên synthetic data
├── test_transitions.py    # Phát hiện chuyển pha đúng vị trí
├── test_dynamics.py       # Velocity/acceleration/cross-corr
├── test_real_data.py      # No NaN, valid range trên BTC thật
├── test_benchmark.py      # So sánh vs hệ thống cũ
└── conftest.py            # Synthetic data generators
```

**Pass criteria**: 9 tiêu chí trong MEASUREMENT_ENGINE.md section 6.

### Phase 2: Edge Discovery

**Tài liệu**: `EDGE_DISCOVERY.md`

**Prerequisite**: Phase 1 pass tất cả test.

| Approach | Phương pháp | Rủi ro | Điểm mạnh |
|----------|-------------|--------|------------|
| A (Data-Driven) | Scan 33 chiều → tìm vùng hit_rate ≠ 50% | Overfitting | Tìm được edge không ngờ |
| B (Theory-Driven) | 5 giả thuyết regime/transition → signals | Miss edge | Stable, ít overfit |
| C (Combined) | B lọc + A tinh chỉnh | Ít signals | Chất lượng cao nhất |

**Files**:
```
strategies/
├── edge_scanner.py        # Approach A
├── theory_signals.py      # Approach B (5 giả thuyết)
└── signal_combiner.py     # Approach C

scripts/
├── run_edge_discovery.py  # Chạy A offline → báo cáo
└── run_ab_test.py         # So sánh A vs B vs C

tests/
├── test_edge_scanner.py
├── test_theory_signals.py
└── test_ab_comparison.py
```

### Phase 3: Strategy Engine (future)

Sau khi Edge Discovery cho ra signal generator thắng cuộc:

- Tích hợp vào NautilusTrader strategy
- Position sizing dựa trên regime + signal strength
- Risk management: CHAOS filter, max drawdown, correlation-based sizing
- Chạy song song với strategy cũ để A/B test

### Phase 4: Backtest & Live (future)

- Full backtest trên NautilusTrader
- So sánh: strategy mới vs strategy cũ vs buy-and-hold
- Paper trading trước live
- Gradual rollout

---

## Thứ tự thực hiện chi tiết

### Phase 1A: Foundation (Metric Engine)
1. `stat_metrics.py` — 6 metric gốc + multi-scale
2. `tests/test_metrics.py` — unit test
3. Run tests → fix → pass

### Phase 1B: State Vector
4. `state_vector.py` — velocity, acceleration, cross-correlation
5. `tests/test_dynamics.py` — test dynamics
6. Run tests → fix → pass

### Phase 1C: Regime
7. `regime_membership.py` — membership + transition scoring
8. `tests/conftest.py` — synthetic data generators
9. `tests/test_regime_synthetic.py` + `tests/test_transitions.py`
10. Run tests → fix → pass

### Phase 1D: Validation
11. `tests/test_robustness.py` — window stability
12. `tests/test_real_data.py` — BTC real data
13. `tests/test_benchmark.py` — vs hệ thống cũ
14. Run all → pass criteria check

### Phase 2A: Approach A (Data-Driven)
15. `edge_scanner.py`
16. `run_edge_discovery.py` → scan BTC data → report
17. `tests/test_edge_scanner.py` → out-of-sample validation

### Phase 2B: Approach B (Theory-Driven)
18. `theory_signals.py` — 5 giả thuyết
19. `tests/test_theory_signals.py` — test từng giả thuyết
20. Keep passing hypotheses, drop failing ones

### Phase 2C: A/B Test
21. `run_ab_test.py` → A vs B trên cùng data
22. Compare all metrics
23. If C wins → `signal_combiner.py`

### Phase 3: Strategy Integration
24. Winner signal generator → `olympex_strategy.py`
25. Full NautilusTrader backtest
26. Compare vs old strategy

---

## Giữ gì từ hệ thống cũ

| File | Hành động | Lý do |
|------|-----------|-------|
| `market_measure.py` | **GIỮ NGUYÊN** | Backward compat + benchmark baseline |
| `regime_classifier.py` | **GIỮ NGUYÊN** | Benchmark baseline |
| `regime_detector.py` | **GIỮ NGUYÊN** | Benchmark baseline |
| `strategy_modules.py` | **GIỮ NGUYÊN** | Có thể tái sử dụng logic entry |
| `olympex_strategy.py` | **SỬA SAU** (Phase 3) | Thêm integration mới |
| `run_backtest.py` | **SỬA SAU** (Phase 3) | Thêm A/B test mode |

**Nguyên tắc**: Không xoá, không sửa cái cũ cho đến khi cái mới CHỨNG MINH tốt hơn.
