# Statistical Measurement Engine

## Triết lý

Thị trường chỉ cung cấp 4 thứ: **Price (OHLC)**, **Volume**, **Time**, **Order Book**.
Mọi thứ khác (RSI, MACD, EMA, ADX, Bollinger Bands...) là biến đổi toán học của giá
với tham số tự chọn. Chúng không sai, nhưng chúng **không phải thuộc tính nội tại**
của thị trường.

Engine này chỉ dùng **thuộc tính thống kê nội tại** — những thứ đúng bất kể
tham số nào được chọn.

---

## 1. Sáu Metric Gốc

| # | Metric | Công thức | Output | Ý nghĩa |
|---|--------|-----------|--------|----------|
| 1 | **Realized Volatility** | `std(returns, window)` | float ≥ 0 | Biến động thật |
| 2 | **Return Autocorrelation** | `corr(r[t], r[t-1])` | -1 to +1 | Quán tính (+) hay đảo ngược (-) |
| 3 | **Skewness** | Moment bậc 3 / σ³ | float | Lệch trái (âm) hay phải (dương) |
| 4 | **Kurtosis** | Moment bậc 4 / σ⁴ | float > 0 | Normal ≈ 3, fat tail >> 3 |
| 5 | **Volume Anomaly** | `volume / median(volume, window)` | float > 0 | Bình thường ≈ 1, spike >> 1 |
| 6 | **Price-Volume Correlation** | `corr(\|return\|, volume)` | -1 to +1 | Volume xác nhận (+) hay mâu thuẫn (-) |

### Tại sao 6 cái này?

1. **Tính trực tiếp** từ returns và volume — không cần indicator trung gian
2. **Ranking ổn định** khi đổi window: vol cao vẫn cao dù đo trên 20 hay 50 bars
3. **Mỗi chiều độc lập**: volatility ≠ autocorrelation ≠ kurtosis
4. **Toán thuần tuý**: không có "magic number" nào (RSI 14? tại sao không 13?)

### Multi-Scale Computation

Mỗi metric được tính trên **nhiều window** rồi tổng hợp:

```python
SCALES = [5, 10, 20, 50, 100]  # bars

# Ví dụ: Realized Volatility
vol_values = [std(returns[-s:]) for s in SCALES if len(returns) >= s]

# Consensus score: bao nhiêu scale đồng ý rằng "cao"?
# Nếu 5/5 scales đều cho vol cao → chắc chắn cao
# Nếu chỉ scale 5 cho cao → mới bắt đầu tăng
```

**Robustness test**: Ranking nhất quán >= 85% trên mọi cặp window → metric đáng tin.

---

## 2. State Vector 33 Chiều

Từ 6 metric gốc, tính thêm dynamics mà KHÔNG cần thêm bất kỳ input mới nào:

### Tầng 0 — VỊ TRÍ (6 số)

```
S(t) = [vol, autocorr, kurtosis, vol_anomaly, pv_corr, skew]
```

"Ta đang ở đâu trên bản đồ." Snapshot hiện tại.

### Tầng 1 — VẬN TỐC (6 số)

```
V(t) = S(t) - S(t-1) = [Δvol, Δautocorr, Δkurtosis, Δvol_anomaly, Δpv_corr, Δskew]
```

"Ta đang đi hướng nào, nhanh cỡ nào."

Ví dụ: Δvol = +0.3 → volatility đang tăng. Δautocorr = -0.1 → momentum đang yếu đi.

### Tầng 2 — GIA TỐC (6 số)

```
A(t) = V(t) - V(t-1) = [ΔΔvol, ΔΔautocorr, ΔΔkurtosis, ΔΔvol_anomaly, ΔΔpv_corr, ΔΔskew]
```

"Ta đang tăng tốc hay đạp phanh?"

Ví dụ: Δvol = +0.3, ΔΔvol = +0.1 → vol tăng VÀ tăng **nhanh hơn** → nguy hiểm.
        Δvol = +0.3, ΔΔvol = -0.1 → vol tăng NHƯNG **chậm lại** → sắp ổn định.

### Tầng 3 — TƯƠNG QUAN CHÉO (15 số)

```
15 cặp = C(6,2):
corr(Δvol, Δautocorr), corr(Δvol, Δkurtosis), corr(Δvol, Δvol_anomaly),
corr(Δvol, Δpv_corr), corr(Δvol, Δskew),
corr(Δautocorr, Δkurtosis), corr(Δautocorr, Δvol_anomaly),
corr(Δautocorr, Δpv_corr), corr(Δautocorr, Δskew),
corr(Δkurtosis, Δvol_anomaly), corr(Δkurtosis, Δpv_corr), corr(Δkurtosis, Δskew),
corr(Δvol_anomaly, Δpv_corr), corr(Δvol_anomaly, Δskew),
corr(Δpv_corr, Δskew)
```

"Các bánh xe có quay cùng hướng không?"

Ví dụ:
- `corr(Δvol, Δkurtosis) = +0.9` → vol tăng kèm fat tail → **CHAOS đang đến**
- `corr(Δvol, Δautocorr) = +0.8` → vol tăng kèm momentum → **MOMENTUM mạnh lên**
- `corr(Δvol_anomaly, Δpv_corr) = -0.7` → volume tăng nhưng không confirm → **Divergence**

### Tổng cộng

```
Tầng 0:  6 số  → Vị trí
Tầng 1:  6 số  → Vận tốc
Tầng 2:  6 số  → Gia tốc
Tầng 3: 15 số  → Tương quan chéo
────────────────
Tổng:   33 số  → 100% tính được bằng toán thuần tuý
```

### Insight quan trọng: Cùng vị trí, khác số phận

```
Thời điểm A: S = [0.5, 0.3, 3.5, 1.2, 0.6, 0.1]   V = [+0.1, +0.05, ...]  → Tiến vào MOMENTUM
Thời điểm B: S = [0.5, 0.3, 3.5, 1.2, 0.6, 0.1]   V = [+0.3, -0.10, ...]  → Lao vào CHAOS

GIỐNG HỆT Tầng 0. KHÁC HOÀN TOÀN Tầng 1.
Nếu chỉ nhìn Tầng 0 = nhìn ảnh chụp.
Biết cả 4 tầng = xem phim.
```

---

## 3. Bốn Regime Thị Trường

### Định nghĩa dựa trên thống kê (không phải nhãn tuỳ ý)

| Regime | Vol | Autocorr | Kurtosis | Vol Anomaly | Bản chất |
|--------|-----|----------|----------|-------------|----------|
| **QUIET** | Thấp | ≈ 0 | Thấp | Thấp | Thị trường ngủ, không ai quan tâm |
| **MOMENTUM** | TB | > 0 (dương) | TB | Tăng | Returns có quán tính — đi tiếp hướng cũ |
| **MEAN-REVERT** | TB | < 0 (âm) | Thấp | Bình thường | Returns đảo ngược — lên rồi xuống |
| **CHAOS** | Cao | Bất kỳ | Cao | Spike | Biến động cực đoan, đuôi béo |

### Tại sao 4 cái này THẬT (không phải bịa):

- **QUIET**: variance nhỏ là mathematical fact
- **MOMENTUM**: autocorrelation dương là mathematical fact
- **MEAN-REVERT**: autocorrelation âm là mathematical fact
- **CHAOS**: kurtosis cao + variance cao là mathematical fact

### Membership liên tục (không nhị phân)

Regime KHÔNG bao giờ nhảy A → B như bật công tắc. Tại mỗi thời điểm:

```
{QUIET: 0.30, MOMENTUM: 0.50, REVERT: 0.15, CHAOS: 0.05}
→ Tổng ≈ 1.0
→ Dominant = MOMENTUM (50%)
→ Secondary = QUIET (30%)
```

### Membership function

```python
def membership(vol, autocorr, kurtosis, vol_anomaly):
    v  = normalize(vol)           # 0 = yên, 1 = loạn
    a  = autocorr                 # -1 đến +1
    k  = normalize(kurtosis)      # 0 = normal, 1 = fat tail
    va = normalize(vol_anomaly)   # 0 = bình thường, 1 = spike

    quiet    = (1-v) * (1-abs(a)) * (1-k) * (1-va)
    momentum = (0.5-abs(v-0.5)) * max(a, 0) * (1-k) * va
    revert   = (0.5-abs(v-0.5)) * max(-a, 0) * (1-k) * (1-va)
    chaos    = v * k * va

    total = quiet + momentum + revert + chaos
    return {name: val/total for name, val in ...}  # normalize to sum=1
```

---

## 4. Chuyển Pha — Bản Đồ & Scoring

### Các đường chuyển hoá tự nhiên

```
                    ┌──────────────────┐
                    │                  │
                    ▼                  │
              ┌──────────┐            │
         ┌───►│  QUIET   ├───────┐    │
         │    └────┬─────┘       │    │
         │         │             │    │ vol↓
         │    vol↑ │ autocorr    │    │ kurtosis↓
         │    dần  │ xuất hiện   │    │
         │         │             │    │
         │    ┌────▼─────┐  ┌───▼────▼───┐
         │    │ MOMENTUM │  │   CHAOS    │
         │    └──┬───┬───┘  └───▲────┬───┘
         │       │   │          │    │
         │       │   │ vol spike│    │ vol↓
         │       │   └──────────┘    │ autocorr flip
         │       │                   │
         │       │ autocorr          │
         │       │ flip âm           │
         │       │                   │
         │    ┌──▼───────────┐       │
         └────┤ MEAN-REVERT  ◄───────┘
              └──────────────┘
```

### Vòng đời phổ biến nhất

```
QUIET → MOMENTUM → MEAN-REVERT → QUIET → ...
                       │
                       ▼ (đôi khi)
                     CHAOS → MOMENTUM → ...
```

### Mỗi chuyển pha có SEQUENCE riêng

**Ví dụ QUIET → MOMENTUM:**

| Bước | Metric thay đổi | Vai trò |
|------|------------------|---------|
| 1 | Volume Anomaly ↑ | Leading — ai đó bắt đầu tích luỹ |
| 2 | Realized Vol ↑ | Early — giá bắt đầu di chuyển |
| 3 | Autocorrelation → dương | Core — quán tính xuất hiện |
| 4 | Price-Volume Corr ↑ | Confirming — volume xác nhận hướng |
| 5 | Skewness lệch rõ | Mature — trend đã establish |

### Transition Score

```
Score = (bước_hoàn_thành / tổng_bước) × 100
      + order_bonus (đúng thứ tự: +10, sai: -10)
      + speed_penalty (quá nhanh: -15)
      + chaos_penalty (-20 × chaos_membership nếu to_regime ≠ CHAOS)
```

### Tốc độ chuyển pha — cũng là signal

| Tốc độ | Bars | Ý nghĩa |
|--------|------|----------|
| Chậm | >20 | Tự nhiên, có thời gian chuẩn bị |
| TB | 5-20 | Bình thường |
| Nhanh | <5 | CHAOS có thể đang đến |

**Bản thân việc regime đổi nhanh CHÍNH LÀ dấu hiệu CHAOS** — bất kể đổi từ gì sang gì.

---

## 5. Hành Động Theo Regime

### QUIET

| Score | Hành động | Size |
|-------|-----------|------|
| 0-50 | Không làm gì | 0% |
| 50-70 | Đặt alert, chuẩn bị | 0% |
| 70-100 | Scale nhỏ, grid nhỏ | 0-30% max |

### MOMENTUM

| Score | Hành động | Size |
|-------|-----------|------|
| 0-30 | Không trade | 0% |
| 30-60 | Entry nhỏ theo hướng autocorr | 30-60% |
| 60-80 | Entry chuẩn + trailing stop | 60-80% |
| 80-100 | Full size + add on pullback | 80-100% |

**Hướng**: Skewness > 0 + autocorr > 0 → LONG. Ngược lại → SHORT.

### MEAN-REVERT

| Score | Hành động | Size |
|-------|-----------|------|
| 0-30 | Không trade | 0% |
| 30-60 | Fade nhẹ tại extreme | 30-60% |
| 60-80 | Fade chuẩn + TP cứng | 60-80% |
| 80-100 | Grid trading | 80-100% |

**Hướng**: Giá > mean + 1σ → SHORT. Giá < mean - 1σ → LONG.

### CHAOS

| Score | Hành động | Size |
|-------|-----------|------|
| 0-30 | Giảm size 50% | ↓50% |
| 30-60 | Giảm size 75% + chặt stop | ↓75% |
| 60-80 | Chỉ hedge, không mở mới | hedge only |
| 80-100 | ĐÓNG HẾT + ĐỨNG NGOÀI | 0% |

### Quy tắc pha trộn

| Rule | Điều kiện | Hành động |
|------|-----------|-----------|
| Dominant > 70% | 1 regime rõ ràng | Theo regime đó |
| Dominant 50-70% | Có secondary | Theo dominant nhưng điều chỉnh theo secondary |
| Không ai > 50% | Thị trường confused | **KHÔNG TRADE** |
| CHAOS > 40% dù không dominant | Thuốc độc | Giảm size × (1 - chaos_weight) |

---

## 6. Test Plan

Xem file riêng hoặc plan file cho chi tiết đầy đủ. Tóm tắt:

| Tầng test | Mục đích | File |
|-----------|----------|------|
| Unit test | Mỗi metric tính đúng | `tests/test_metrics.py` |
| Robustness | Ranking stable khi đổi window | `tests/test_robustness.py` |
| Regime synthetic | Classify đúng trên data biết trước | `tests/test_regime_synthetic.py` |
| Transition | Phát hiện chuyển pha đúng | `tests/test_transitions.py` |
| Dynamics | Velocity/accel/cross-corr đúng | `tests/test_dynamics.py` |
| Real data | Không NaN, range hợp lý | `tests/test_real_data.py` |
| Benchmark | So sánh với hệ thống cũ | `tests/test_benchmark.py` |
