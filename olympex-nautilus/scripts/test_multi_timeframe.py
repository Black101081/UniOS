#!/usr/bin/env python3
"""
Multi-Timeframe Edge Test
=========================
Test cùng formula trên M1, M15, H1 data.
H1 có 6 tháng data (bull + bear) → sample lớn hơn nhiều.

Mục tiêu:
1. Xác nhận pattern có tồn tại trên nhiều timeframe không
2. Tính target X tối ưu theo ATR từng timeframe
3. Tính số lệnh trung bình / ngày
4. Xác định base volume threshold
5. Kiểm tra xem pattern có hoạt động ở cả LONG và SHORT không
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")

SCALES = [5, 10, 20, 50, 100]
DATA_DIR = Path(__file__).parent.parent / "data"


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
    return states


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


def walk_forward_test(closes, states, target, max_bars, condition_func, n_folds=5):
    n = len(closes)
    labels = label_moves(closes, target, max_bars)
    valid_indices = [i for i in range(n) if states[i] is not None and labels[i] != 0]
    if len(valid_indices) < n_folds * 5:
        return None

    fold_size = len(valid_indices) // n_folds
    fold_results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < n_folds - 1 else len(valid_indices)
        test_indices = valid_indices[start:end]
        matches = [(i, labels[i]) for i in test_indices if condition_func(states[i])]
        if len(matches) < 2:
            fold_results.append({"fold": fold, "n": 0, "n_up": 0, "n_down": 0, "hr_up": 0.5})
            continue
        n_up = sum(1 for _, l in matches if l == 1)
        fold_results.append({
            "fold": fold, "n": len(matches),
            "n_up": n_up, "n_down": len(matches) - n_up,
            "hr_up": n_up / len(matches),
        })

    total_n = sum(f["n"] for f in fold_results)
    total_up = sum(f["n_up"] for f in fold_results)
    if total_n == 0:
        return None

    overall_hr_up = total_up / total_n
    direction = 1 if overall_hr_up > 0.5 else -1
    directional_hr = overall_hr_up if direction == 1 else (1 - overall_hr_up)

    fold_hrs = []
    for f in fold_results:
        if f["n"] > 0:
            hr = f["hr_up"] if direction == 1 else (1 - f["hr_up"])
            fold_hrs.append(hr)

    consistent = sum(1 for h in fold_hrs if h > 0.5)
    active = len(fold_hrs)

    try:
        p = stats.binomtest(
            total_up if direction == 1 else (total_n - total_up),
            total_n, 0.5).pvalue
    except:
        p = 1.0

    return {
        "total_n": total_n, "total_up": total_up, "total_down": total_n - total_up,
        "direction": direction, "directional_hr": directional_hr,
        "p_value": p, "consistent": consistent, "active": active,
        "fold_hrs": fold_hrs, "fold_details": fold_results,
    }


# ── CONDITIONS (percentile-adaptive) ──

def make_adaptive_conditions(states):
    """
    Instead of hardcoded thresholds, use percentile-based.
    This adapts to different timeframes automatically.
    """
    # Collect distributions
    vol_vals = [s["vol"] for s in states if s is not None]
    pv_vals = [s.get("pv_corr", 0) for s in states if s is not None]
    kurt_vals = [s.get("kurt", 3) for s in states if s is not None]
    xc_sp_vals = [s.get("xc_skew_pv_corr", 0) for s in states if s is not None]
    xc_va_vals = [s.get("xc_vol_autocorr", 0) for s in states if s is not None]
    xc_vk_vals = [s.get("xc_vol_kurt", 0) for s in states if s is not None]
    ac_vals = [s.get("autocorr", 0) for s in states if s is not None]

    if not vol_vals:
        return {}

    # Compute percentile thresholds
    vol_p10 = np.percentile(vol_vals, 10)
    vol_p20 = np.percentile(vol_vals, 20)
    vol_p5 = np.percentile(vol_vals, 5)
    pv_p25 = np.percentile(pv_vals, 25)
    kurt_p50 = np.percentile(kurt_vals, 50)
    xc_sp_p15 = np.percentile(xc_sp_vals, 15)
    xc_sp_p25 = np.percentile(xc_sp_vals, 25)
    xc_va_p30 = np.percentile(xc_va_vals, 30)
    xc_vk_p20 = np.percentile(xc_vk_vals, 20)
    ac_p20 = np.percentile(ac_vals, 20)
    ac_p80 = np.percentile(ac_vals, 80)

    print(f"    Adaptive thresholds:")
    print(f"      vol_p10={vol_p10:.6f} vol_p20={vol_p20:.6f}")
    print(f"      xc_skew_pv_p15={xc_sp_p15:.4f} xc_skew_pv_p25={xc_sp_p25:.4f}")
    print(f"      pv_corr_p25={pv_p25:.4f}")
    print(f"      kurt_p50={kurt_p50:.4f}")
    print(f"      autocorr_p20={ac_p20:.4f} autocorr_p80={ac_p80:.4f}")

    conditions = {
        # P1 equivalent: very low vol + negative skew-pv cross-corr
        "P1_quiet_storm": lambda s: s.get("vol", 999) <= vol_p10 and s.get("xc_skew_pv_corr", 0) <= xc_sp_p15,

        # P2 equivalent: low vol + negative vol-autocorr cross-corr
        "P2_quiet_decouple": lambda s: s.get("vol", 999) <= vol_p10 and s.get("xc_vol_autocorr", 0) <= xc_va_p30,

        # P3 equivalent: low kurtosis + low vol
        "P3_normal_quiet": lambda s: s.get("kurt", 999) <= kurt_p50 and s.get("vol", 999) <= vol_p10,

        # P4 equivalent: just low vol
        "P4_just_quiet": lambda s: s.get("vol", 999) <= vol_p10,

        # P5 equivalent: low vol + low pv correlation
        "P5_quiet_no_confirm": lambda s: s.get("vol", 999) <= vol_p10 and s.get("pv_corr", 999) <= pv_p25,

        # NEW: Momentum — high autocorr + positive skew (should predict LONG)
        "P8_momentum_up": lambda s: s.get("autocorr", 0) >= ac_p80 and s.get("skew", 0) > 0,

        # NEW: Momentum — high autocorr + negative skew (should predict SHORT)
        "P9_momentum_down": lambda s: s.get("autocorr", 0) >= ac_p80 and s.get("skew", 0) < 0,

        # NEW: High vol + high kurtosis = chaos → should predict DOWN or be useless
        "P10_chaos": lambda s: s.get("vol", 0) >= np.percentile(vol_vals, 90) and s.get("kurt", 0) >= np.percentile(kurt_vals, 80),
    }

    return conditions


def analyze_timeframe(name, filepath, candle_minutes):
    print(f"\n{'='*90}")
    print(f"TIMEFRAME: {name} | Candle: {candle_minutes}min")
    print(f"{'='*90}")

    df = pd.read_parquet(filepath).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)

    # Basic stats
    avg_price = closes.mean()
    avg_move = np.mean(np.abs(np.diff(closes)))
    std_move = np.std(np.diff(closes))
    price_range = closes.max() - closes.min()

    # Time span
    ts_start = df["timestamp"].iloc[0]
    ts_end = df["timestamp"].iloc[-1]
    days_span = (pd.to_datetime(ts_end) - pd.to_datetime(ts_start)).total_seconds() / 86400

    print(f"  Candles: {n:,} | Days: {days_span:.1f}")
    print(f"  Price: ${closes.min():,.0f} → ${closes.max():,.0f} (range ${price_range:,.0f})")
    print(f"  Avg price: ${avg_price:,.0f}")
    print(f"  Avg candle move: ${avg_move:.1f} | Std: ${std_move:.1f}")
    print(f"  1 ATR ≈ ${avg_move:.0f}")

    # Volume stats
    avg_vol = volumes.mean()
    med_vol = np.median(volumes)
    print(f"  Avg volume: {avg_vol:.4f} BTC | Median: {med_vol:.4f} BTC")
    print(f"  Avg $ volume/candle: ${avg_vol * avg_price:,.0f}")

    # Build states
    print(f"\n  Building states...")
    t0 = time.time()
    states = build_states(closes, volumes)
    valid = sum(1 for s in states if s is not None)
    print(f"  {valid:,} valid states ({time.time()-t0:.1f}s)")

    # Adaptive conditions
    print()
    conditions = make_adaptive_conditions(states)
    if not conditions:
        print("  ERROR: No conditions generated")
        return

    # Target sweep: use ATR-multiples
    atr = avg_move
    target_multipliers = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30]
    targets = [(round(atr * m, 0), max(30, int(m * 10))) for m in target_multipliers]
    # Deduplicate and limit
    seen = set()
    unique_targets = []
    for t, mb in targets:
        if t not in seen and t > 0 and mb < n * 0.25:
            seen.add(t)
            unique_targets.append((t, mb))

    # Test each condition × target
    print(f"\n  Testing {len(conditions)} conditions × {len(unique_targets)} targets...")
    print()

    # Collect results
    all_results = []

    for cond_name, cond_func in conditions.items():
        # Count triggers
        trigger_count = sum(1 for i in range(n) if states[i] is not None and cond_func(states[i]))
        triggers_per_day = trigger_count / max(days_span, 1)

        best_hr = 0
        best_target = 0
        best_result = None

        for target, max_bars in unique_targets:
            result = walk_forward_test(closes, states, target, max_bars, cond_func, n_folds=5)
            if result is None:
                continue

            if result["directional_hr"] > best_hr and result["total_n"] >= 10:
                best_hr = result["directional_hr"]
                best_target = target
                best_result = result

            if result["total_n"] >= 10 and result["p_value"] < 0.05:
                all_results.append({
                    "condition": cond_name,
                    "target": target,
                    "max_bars": max_bars,
                    "direction": "UP" if result["direction"] == 1 else "DN",
                    "hr": result["directional_hr"],
                    "n": result["total_n"],
                    "p": result["p_value"],
                    "consistent": result["consistent"],
                    "active": result["active"],
                    "fold_hrs": result["fold_hrs"],
                    "triggers_per_day": triggers_per_day,
                    "atr_multiple": target / atr,
                })

        if best_result and best_result["total_n"] >= 10:
            dir_str = "UP" if best_result["direction"] == 1 else "DN"
            fold_str = " ".join(f"{h:.0%}" for h in best_result["fold_hrs"])
            sig = "***" if best_result["p_value"] < 0.001 else "**" if best_result["p_value"] < 0.01 else "*" if best_result["p_value"] < 0.05 else ""
            print(f"  {cond_name:25s} best=${best_target:>6.0f} ({best_target/atr:.1f}ATR) "
                  f"{dir_str} HR={best_hr:.1%} n={best_result['total_n']:>4} p={best_result['p_value']:.4f}{sig} "
                  f"[{fold_str}] trig={triggers_per_day:.1f}/day")

    # Print detailed table of significant results
    sig_results = [r for r in all_results if r["p"] < 0.05 and r["hr"] >= 0.60]
    if sig_results:
        sig_results.sort(key=lambda x: x["hr"], reverse=True)
        print(f"\n  SIGNIFICANT RESULTS (p<0.05, HR>=60%):")
        print(f"  {'Condition':25s} {'Target':>8} {'ATR×':>5} {'Dir':>4} {'HR':>6} {'N':>5} {'p':>8} {'Consist':>8} {'Trig/day':>9} | Folds")
        print(f"  {'─'*100}")
        for r in sig_results[:20]:
            fold_str = " ".join(f"{h:.0%}" for h in r["fold_hrs"])
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
            print(f"  {r['condition']:25s} ${r['target']:>6.0f} {r['atr_multiple']:>4.1f}× {r['direction']:>4} "
                  f"{r['hr']:>5.1%} {r['n']:>5} {r['p']:>7.4f}{sig:>1} "
                  f"{r['consistent']}/{r['active']:>1}      {r['triggers_per_day']:>8.1f} | [{fold_str}]")

    # Summary stats
    if all_results:
        print(f"\n  SUMMARY:")
        print(f"    Total significant conditions (p<0.05): {len([r for r in all_results if r['p'] < 0.05])}")
        print(f"    HR >= 80%: {len([r for r in all_results if r['hr'] >= 0.80])}")
        print(f"    HR >= 70%: {len([r for r in all_results if r['hr'] >= 0.70])}")
        print(f"    HR >= 60%: {len([r for r in all_results if r['hr'] >= 0.60])}")

        # Best overall
        best = max(all_results, key=lambda x: x["hr"])
        print(f"    Best: {best['condition']} ${best['target']:.0f} → {best['hr']:.1%} "
              f"({best['n']} signals, {best['direction']}, {best['triggers_per_day']:.1f}/day)")

    return all_results


def main():
    timeframes = []

    # M1
    m1_path = DATA_DIR / "btc_candles_1min.parquet"
    if m1_path.exists():
        timeframes.append(("M1 (1-minute)", m1_path, 1))

    # M15
    m15_path = DATA_DIR / "btc_candles_15min.parquet"
    if m15_path.exists():
        timeframes.append(("M15 (15-minute)", m15_path, 15))

    # H1
    h1_path = DATA_DIR / "btc_candles_1h.parquet"
    if h1_path.exists():
        timeframes.append(("H1 (1-hour)", h1_path, 60))

    all_tf_results = {}
    for name, path, minutes in timeframes:
        results = analyze_timeframe(name, path, minutes)
        if results:
            all_tf_results[name] = results

    # Cross-timeframe comparison
    print()
    print("=" * 90)
    print("CROSS-TIMEFRAME COMPARISON")
    print("=" * 90)
    print()

    for tf_name, results in all_tf_results.items():
        sig = [r for r in results if r["p"] < 0.05 and r["hr"] >= 0.65]
        if sig:
            best = max(sig, key=lambda x: x["hr"])
            print(f"  {tf_name:25s}: Best={best['condition']} ${best['target']:.0f} "
                  f"→ {best['hr']:.1%} ({best['n']} signals, {best['triggers_per_day']:.1f}/day)")

    # Common patterns across timeframes
    print()
    print("  Patterns that work across timeframes:")
    all_conditions = set()
    for results in all_tf_results.values():
        for r in results:
            if r["p"] < 0.05 and r["hr"] >= 0.60:
                all_conditions.add(r["condition"])

    for cond in sorted(all_conditions):
        tfs_with_edge = []
        for tf_name, results in all_tf_results.items():
            matches = [r for r in results if r["condition"] == cond and r["p"] < 0.05 and r["hr"] >= 0.60]
            if matches:
                best = max(matches, key=lambda x: x["hr"])
                tfs_with_edge.append(f"{tf_name.split()[0]}({best['hr']:.0%})")
        if len(tfs_with_edge) >= 2:
            print(f"    {cond:25s}: {', '.join(tfs_with_edge)}")


if __name__ == "__main__":
    main()
