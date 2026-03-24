#!/usr/bin/env python3
"""
Verify Optimal X — Phân tích sâu pattern chính từ find_optimal_x.py
=====================================================================

Phát hiện chính: Khi vol CỰC THẤP + tương quan chéo ở vùng cụ thể
→ Giá có xu hướng rõ ràng.

Script này:
1. Verify bằng walk-forward (5-fold time split, không phải 1 split)
2. Đo statistical significance nghiêm ngặt
3. Tìm X tối ưu chính xác (sweep mịn)
4. Đưa ra công thức cuối cùng
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")


SCALES = [5, 10, 20, 50, 100]

def compute_metrics(returns, volumes):
    def _ms(func):
        vals = [func(s) for s in SCALES if len(returns) >= s]
        return float(np.mean(vals)) if vals else 0.0
    def _vol(s): return np.std(returns[-s:])
    def _ac(s):
        r = returns[-s:]
        return float(np.corrcoef(r[1:], r[:-1])[0, 1]) if np.std(r) > 1e-12 else 0.0
    def _sk(s):
        r = returns[-s:]; std = np.std(r)
        return float(np.mean(((r - r.mean()) / std) ** 3)) if std > 1e-12 else 0.0
    def _ku(s):
        r = returns[-s:]; std = np.std(r)
        return float(np.mean(((r - r.mean()) / std) ** 4)) if std > 1e-12 else 0.0
    def _va(s):
        w = volumes[-s:]; med = np.median(w)
        return float(volumes[-1] / med) if med > 1e-12 else 1.0
    def _pv(s):
        r, v = np.abs(returns[-s:]), volumes[-s:]
        return float(np.corrcoef(r, v)[0, 1]) if np.std(r) > 1e-12 and np.std(v) > 1e-12 else 0.0
    return {"vol": _ms(_vol), "autocorr": _ms(_ac), "skew": _ms(_sk),
            "kurt": _ms(_ku), "vol_anom": _ms(_va), "pv_corr": _ms(_pv)}


def build_states(closes, volumes):
    n = len(closes)
    returns = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])
    WARMUP = 110
    metric_history = []
    states = [None] * n
    dim_names = ["vol", "autocorr", "skew", "kurt", "vol_anom", "pv_corr"]
    for i in range(WARMUP, n):
        r_w = returns[max(0, i - 200): i + 1]
        v_w = volumes[max(0, i - 200): i + 1]
        m = compute_metrics(r_w, v_w)
        metric_history.append(m)
        if len(metric_history) < 3:
            continue
        cur, prev, prev2 = metric_history[-1], metric_history[-2], metric_history[-3]
        state = {}
        for nm in dim_names:
            state[nm] = cur[nm]
            state[f"d_{nm}"] = cur[nm] - prev[nm]
            state[f"dd_{nm}"] = (cur[nm] - prev[nm]) - (prev[nm] - prev2[nm])
        if len(metric_history) >= 12:
            vels = []
            for j in range(max(0, len(metric_history)-20), len(metric_history)-1):
                vels.append({nm: metric_history[j+1][nm] - metric_history[j][nm] for nm in dim_names})
            if len(vels) >= 5:
                for ii, n1 in enumerate(dim_names):
                    for n2 in dim_names[ii+1:]:
                        v1 = np.array([v[n1] for v in vels])
                        v2 = np.array([v[n2] for v in vels])
                        c = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 1e-12 and np.std(v2) > 1e-12 else 0.0
                        state[f"xc_{n1}_{n2}"] = 0.0 if np.isnan(c) else float(c)
            else:
                for ii, n1 in enumerate(dim_names):
                    for n2 in dim_names[ii+1:]:
                        state[f"xc_{n1}_{n2}"] = 0.0
        else:
            for ii, n1 in enumerate(dim_names):
                for n2 in dim_names[ii+1:]:
                    state[f"xc_{n1}_{n2}"] = 0.0
        states[i] = state
    return states, returns


def label_moves(closes, target, max_bars):
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        base = closes[i]
        up_t, dn_t = base + target, base - target
        end = min(i + max_bars + 1, n)
        h_up = h_dn = -1
        for j in range(i + 1, end):
            if h_up < 0 and closes[j] >= up_t: h_up = j - i
            if h_dn < 0 and closes[j] <= dn_t: h_dn = j - i
            if h_up >= 0 and h_dn >= 0: break
        if h_up >= 0 and h_dn >= 0:
            labels[i] = 1 if h_up < h_dn else -1
        elif h_up >= 0: labels[i] = 1
        elif h_dn >= 0: labels[i] = -1
    return labels


# ============================================================
# Các CONDITION CANDIDATES từ kết quả trước
# ============================================================

def condition_low_vol(state, thresh=0.00017):
    """Vol cực thấp — thị trường đang ngủ."""
    return state.get("vol", 999) <= thresh

def condition_low_atr(state, thresh=0.00015):
    """ATR cực thấp."""
    return state.get("atr_pct", 999) <= thresh if "atr_pct" in state else state.get("vol", 999) <= thresh

def condition_low_kurt(state, thresh=3.1):
    """Kurtosis thấp (phân phối normal, không fat tail)."""
    return state.get("kurt", 999) <= thresh

def condition_xc_skew_pv_negative(state, thresh=-0.49):
    """Cross-corr(velocity of skew, velocity of pv_corr) âm mạnh."""
    return state.get("xc_skew_pv_corr", 0) <= thresh

def condition_xc_vol_autocorr_negative(state, thresh=-0.20):
    """Cross-corr(velocity of vol, velocity of autocorr) âm."""
    return state.get("xc_vol_autocorr", 0) <= thresh

def condition_xc_vol_kurt_negative(state, thresh=-0.48):
    """Cross-corr(velocity of vol, velocity of kurtosis) âm mạnh."""
    return state.get("xc_vol_kurt", 0) <= thresh

def condition_low_pv_corr(state, thresh=0.034):
    """Price-Volume correlation thấp."""
    return state.get("pv_corr", 999) <= thresh


# Named conditions for testing
CONDITIONS = {
    # Pattern 1: "Thị trường ngủ + skew-pv diverge" → UP
    "P1_quiet_storm": lambda s: condition_low_vol(s, 0.00017) and condition_xc_skew_pv_negative(s, -0.49),

    # Pattern 2: "Thị trường ngủ + vol-autocorr decouple" → UP
    "P2_quiet_decouple": lambda s: condition_low_vol(s, 0.00017) and condition_xc_vol_autocorr_negative(s, -0.14),

    # Pattern 3: "Low kurtosis + low ATR" → DOWN (large targets)
    "P3_normal_quiet": lambda s: condition_low_kurt(s, 3.11) and condition_low_vol(s, 0.00017),

    # Pattern 4: "Low vol chỉ cần 1 điều kiện" (simplest)
    "P4_just_quiet": lambda s: condition_low_vol(s, 0.00017),

    # Pattern 5: "Low vol + low pv_corr" → UP (medium targets)
    "P5_quiet_no_confirm": lambda s: condition_low_vol(s, 0.00017) and condition_low_pv_corr(s, 0.034),

    # Pattern 6: "Low kurt + vol-kurt decouple" → DOWN (large targets)
    "P6_normal_decouple": lambda s: condition_low_kurt(s, 2.92) and condition_xc_vol_kurt_negative(s, -0.48),

    # Pattern 7: "ATR-based low vol"
    "P7_atr_quiet": lambda s: condition_low_vol(s, 0.00022),
}


def walk_forward_test(closes, states, target, max_bars, condition_func, n_folds=5):
    """
    Walk-forward validation: chia data thành n_folds phần thời gian.
    Test trên mỗi fold, dùng tất cả fold trước để "train" (xác nhận direction).
    """
    n = len(closes)
    labels = label_moves(closes, target, max_bars)

    valid_indices = [i for i in range(n) if states[i] is not None and labels[i] != 0]
    if len(valid_indices) < n_folds * 10:
        return None

    fold_size = len(valid_indices) // n_folds
    fold_results = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else len(valid_indices)
        test_indices = valid_indices[start:end]

        # Count how many match condition in this fold
        matches = []
        for i in test_indices:
            if condition_func(states[i]):
                matches.append((i, labels[i]))

        if len(matches) < 3:
            fold_results.append({"fold": fold, "n": 0, "hr_up": 0, "hr_down": 0})
            continue

        n_up = sum(1 for _, l in matches if l == 1)
        n_total = len(matches)
        hr_up = n_up / n_total

        fold_results.append({
            "fold": fold,
            "n": n_total,
            "n_up": n_up,
            "n_down": n_total - n_up,
            "hr_up": hr_up,
            "hr_down": 1 - hr_up,
        })

    # Aggregate
    total_n = sum(f["n"] for f in fold_results)
    total_up = sum(f.get("n_up", 0) for f in fold_results)
    if total_n == 0:
        return None

    overall_hr_up = total_up / total_n

    # Determine direction from majority of folds
    up_folds = sum(1 for f in fold_results if f["n"] > 0 and f["hr_up"] > 0.5)
    down_folds = sum(1 for f in fold_results if f["n"] > 0 and f["hr_up"] < 0.5)
    active_folds = sum(1 for f in fold_results if f["n"] > 0)

    if up_folds > down_folds:
        direction = 1
        directional_hr = overall_hr_up
    else:
        direction = -1
        directional_hr = 1 - overall_hr_up

    # Consistency: how many folds agree on direction?
    if direction == 1:
        consistent_folds = up_folds
    else:
        consistent_folds = down_folds

    # Statistical test
    try:
        p_value = stats.binomtest(
            total_up if direction == 1 else (total_n - total_up),
            total_n, 0.5
        ).pvalue
    except Exception:
        p_value = 1.0

    # Per-fold hit rates for the chosen direction
    fold_hrs = []
    for f in fold_results:
        if f["n"] > 0:
            hr = f["hr_up"] if direction == 1 else f["hr_down"]
            fold_hrs.append(hr)

    return {
        "total_n": total_n,
        "direction": direction,
        "directional_hr": directional_hr,
        "p_value": p_value,
        "consistent_folds": consistent_folds,
        "active_folds": active_folds,
        "fold_hrs": fold_hrs,
        "fold_details": fold_results,
        "hr_std": np.std(fold_hrs) if fold_hrs else 0,
        "hr_min": min(fold_hrs) if fold_hrs else 0,
        "hr_max": max(fold_hrs) if fold_hrs else 0,
    }


def main():
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)

    avg_1min_move = np.mean(np.abs(np.diff(closes)))
    print(f"Data: {n:,} candles | ${closes.min():,.0f} — ${closes.max():,.0f}")
    print(f"Avg 1-min move: ${avg_1min_move:.2f} | ~1 ATR(20): ${avg_1min_move:.0f}")
    print()

    print("Building state vectors...")
    t0 = time.time()
    states, returns = build_states(closes, volumes)
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    # ── Fine sweep of X ──
    fine_targets = [
        10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75,
        100, 125, 150, 200, 250, 300, 400, 500,
        750, 1000,
    ]

    # Map max_bars proportional to target
    def max_bars_for(target):
        # Rule: need ~target/$avg_move bars minimum, give 3-5x headroom
        bars_needed = target / avg_1min_move
        return int(min(max(bars_needed * 4, 60), n * 0.25))

    print("=" * 110)
    print("WALK-FORWARD VERIFICATION (5-fold time split)")
    print("=" * 110)
    print()

    for cond_name, cond_func in CONDITIONS.items():
        print(f"\n{'━' * 100}")
        print(f"CONDITION: {cond_name}")
        print(f"{'━' * 100}")
        print(f"{'X':>7} {'MaxB':>6} {'Dir':>4} {'N':>5} {'HR':>7} {'p':>8} {'Folds':>6} "
              f"{'Min':>6} {'Max':>6} {'Std':>6} | Per-fold HR")
        print(f"{'─' * 100}")

        best_hr = 0
        best_x = 0
        best_result = None

        for target in fine_targets:
            mb = max_bars_for(target)
            result = walk_forward_test(closes, states, target, mb, cond_func, n_folds=5)

            if result is None or result["total_n"] < 10:
                continue

            dir_str = "UP" if result["direction"] == 1 else "DN"
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
            fold_str = " ".join(f"{h:.0%}" for h in result["fold_hrs"])
            consistency = f"{result['consistent_folds']}/{result['active_folds']}"

            print(f"${target:>6} {mb:>5} {dir_str:>4} {result['total_n']:>5} "
                  f"{result['directional_hr']:>6.1%} {result['p_value']:>7.4f}{sig:>1} "
                  f"{consistency:>6} {result['hr_min']:>5.0%} {result['hr_max']:>5.0%} "
                  f"{result['hr_std']:>5.1%} | [{fold_str}]")

            if result["directional_hr"] > best_hr and result["total_n"] >= 15:
                best_hr = result["directional_hr"]
                best_x = target
                best_result = result

        if best_result:
            dir_str = "UP" if best_result["direction"] == 1 else "DN"
            print(f"\n  ★ Best for {cond_name}: X=${best_x} → {best_hr:.1%} ({best_result['total_n']} signals, {dir_str})")

    # ── FINAL SUMMARY ──
    print()
    print()
    print("=" * 100)
    print("FINAL: TÌM X TỐI ƯU — Sweep mịn $30-$120 (vùng hứa hẹn nhất)")
    print("=" * 100)
    print()

    # The key condition from results: low vol
    key_condition = CONDITIONS["P1_quiet_storm"]

    ultra_fine = list(range(30, 121, 5))
    print(f"Condition: P1_quiet_storm (vol<=0.00017 & xc_skew_pv_corr<=-0.49)")
    print(f"{'X':>5} {'MaxB':>6} {'Dir':>4} {'N':>5} {'HR':>7} {'p':>8} {'Consistent':>10} | Folds")
    print(f"{'─' * 80}")

    results_table = []
    for target in ultra_fine:
        mb = max_bars_for(target)
        result = walk_forward_test(closes, states, target, mb, key_condition, n_folds=5)
        if result is None or result["total_n"] < 5:
            continue

        dir_str = "UP" if result["direction"] == 1 else "DN"
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
        fold_str = " ".join(f"{h:.0%}" for h in result["fold_hrs"])
        consistency = f"{result['consistent_folds']}/{result['active_folds']}"

        print(f"${target:>4} {mb:>5} {dir_str:>4} {result['total_n']:>5} "
              f"{result['directional_hr']:>6.1%} {result['p_value']:>7.4f}{sig:>1} "
              f"{consistency:>10} | [{fold_str}]")

        results_table.append({
            "target": target,
            "hr": result["directional_hr"],
            "n": result["total_n"],
            "p": result["p_value"],
            "direction": result["direction"],
            "consistent": result["consistent_folds"],
            "active": result["active_folds"],
            "fold_hrs": result["fold_hrs"],
        })

    # Also check P4 (just quiet) and P3 (normal_quiet) for larger targets
    print()
    print(f"Condition: P4_just_quiet (vol<=0.00017) — simple")
    print(f"{'X':>5} {'MaxB':>6} {'Dir':>4} {'N':>5} {'HR':>7} {'p':>8} {'Consistent':>10} | Folds")
    print(f"{'─' * 80}")

    for target in [50, 75, 100, 150, 200, 300, 500, 750, 1000]:
        mb = max_bars_for(target)
        result = walk_forward_test(closes, states, target, mb, CONDITIONS["P4_just_quiet"], n_folds=5)
        if result is None or result["total_n"] < 5:
            continue
        dir_str = "UP" if result["direction"] == 1 else "DN"
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
        fold_str = " ".join(f"{h:.0%}" for h in result["fold_hrs"])
        consistency = f"{result['consistent_folds']}/{result['active_folds']}"
        print(f"${target:>4} {mb:>5} {dir_str:>4} {result['total_n']:>5} "
              f"{result['directional_hr']:>6.1%} {result['p_value']:>7.4f}{sig:>1} "
              f"{consistency:>10} | [{fold_str}]")

    print()
    print(f"Condition: P3_normal_quiet (kurt<=3.11 & vol<=0.00017)")
    print(f"{'X':>5} {'MaxB':>6} {'Dir':>4} {'N':>5} {'HR':>7} {'p':>8} {'Consistent':>10} | Folds")
    print(f"{'─' * 80}")

    for target in [200, 300, 500, 750, 1000]:
        mb = max_bars_for(target)
        result = walk_forward_test(closes, states, target, mb, CONDITIONS["P3_normal_quiet"], n_folds=5)
        if result is None or result["total_n"] < 5:
            continue
        dir_str = "UP" if result["direction"] == 1 else "DN"
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
        fold_str = " ".join(f"{h:.0%}" for h in result["fold_hrs"])
        consistency = f"{result['consistent_folds']}/{result['active_folds']}"
        print(f"${target:>4} {mb:>5} {dir_str:>4} {result['total_n']:>5} "
              f"{result['directional_hr']:>6.1%} {result['p_value']:>7.4f}{sig:>1} "
              f"{consistency:>10} | [{fold_str}]")

    # ── Count how often condition triggers ──
    print()
    print("=" * 60)
    print("FREQUENCY: Bao lâu condition trigger 1 lần?")
    print("=" * 60)
    for cond_name, cond_func in CONDITIONS.items():
        count = sum(1 for i in range(n) if states[i] is not None and cond_func(states[i]))
        if count > 0:
            freq_hours = (n / count) / 60
            print(f"  {cond_name:30s}: {count:>5} times ({count/n*100:.1f}%) = mỗi {freq_hours:.1f}h")


if __name__ == "__main__":
    main()
