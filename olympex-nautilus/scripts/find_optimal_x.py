#!/usr/bin/env python3
"""
Tìm X Tối Ưu — Ở mức nào thì dự đoán đúng nhiều nhất?
========================================================

Bài toán: Cho mỗi bar t, giá sẽ tăng $X hay giảm $X trước?
Tìm: X nào cho hit rate cao nhất, với điều kiện nào?

Approach:
1. Sweep X từ $1 → $2000
2. Với mỗi X, tìm SUBSET điều kiện cho hit rate cao nhất
3. Ràng buộc: minimum 30 signals (không phải overfitting trên 3 samples)
4. Out-of-sample validation bắt buộc
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")


# ── Measurement Engine (compact) ──

SCALES = [5, 10, 20, 50, 100]

def compute_metrics(returns, volumes):
    """6 metrics, multi-scale consensus."""
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

    return {
        "vol": _ms(_vol), "autocorr": _ms(_ac), "skew": _ms(_sk),
        "kurt": _ms(_ku), "vol_anom": _ms(_va), "pv_corr": _ms(_pv),
    }


def build_states(closes, volumes):
    """Build state vectors for all bars."""
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

        # Cross-correlations (use last 20 velocity samples)
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

        # Extra: recent price momentum (raw)
        if i >= 20:
            state["ret_20"] = float((closes[i] - closes[i-20]) / closes[i-20])
            state["ret_5"] = float((closes[i] - closes[i-5]) / closes[i-5])
        else:
            state["ret_20"] = 0.0
            state["ret_5"] = 0.0

        # ATR proxy (avg absolute return × price)
        atr_window = min(20, len(r_w) - 1)
        if atr_window > 0:
            state["atr_pct"] = float(np.mean(np.abs(r_w[-atr_window:])))
        else:
            state["atr_pct"] = 0.0

        states[i] = state

    return states


def label_moves(closes, target, max_bars):
    """Label: +1 if up $X first, -1 if down $X first, 0 if neither."""
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    bars_to_hit = np.zeros(n, dtype=np.int32)  # how many bars to reach target

    for i in range(n - 1):
        base = closes[i]
        up_t = base + target
        dn_t = base - target
        end = min(i + max_bars + 1, n)

        h_up = h_dn = -1
        for j in range(i + 1, end):
            if h_up < 0 and closes[j] >= up_t:
                h_up = j - i
            if h_dn < 0 and closes[j] <= dn_t:
                h_dn = j - i
            if h_up >= 0 and h_dn >= 0:
                break

        if h_up >= 0 and h_dn >= 0:
            labels[i] = 1 if h_up < h_dn else -1
            bars_to_hit[i] = min(h_up, h_dn)
        elif h_up >= 0:
            labels[i] = 1
            bars_to_hit[i] = h_up
        elif h_dn >= 0:
            labels[i] = -1
            bars_to_hit[i] = h_dn

    return labels, bars_to_hit


def find_best_conditions(states, labels, mask, min_samples=30, max_dim_combos=2):
    """
    Tìm điều kiện cho hit rate cao nhất.
    Returns list of (condition_desc, hit_rate, count, direction, filter_func)
    """
    valid = (labels != 0) & mask
    if valid.sum() < min_samples:
        return []

    sample = None
    for i in range(len(states)):
        if states[i] is not None and valid[i]:
            sample = states[i]
            break
    if not sample:
        return []

    dim_names = sorted(sample.keys())
    n = len(states)

    # Pre-extract all dimension values
    dim_vals = {}
    for d in dim_names:
        dim_vals[d] = np.array([states[i].get(d, 0) if states[i] else 0 for i in range(n)])

    # Step 1: Find single-dim predictive power
    single_results = []
    for d in dim_names:
        dv = dim_vals[d][valid]
        dl = labels[valid]
        if np.std(dv) < 1e-15:
            continue

        # Try multiple thresholds
        for pct in [10, 20, 25, 33, 50, 67, 75, 80, 90]:
            thresh = np.percentile(dv, pct)

            # Above threshold
            above = dv > thresh
            if above.sum() >= min_samples // 2:
                hr = (dl[above] == 1).mean()
                single_results.append({
                    "dims": [d], "ops": [">"], "thresholds": [float(thresh)],
                    "hr": hr, "n": int(above.sum()), "pct": pct,
                    "edge": abs(hr - 0.5)
                })

            # Below threshold
            below = dv <= thresh
            if below.sum() >= min_samples // 2:
                hr = (dl[below] == 1).mean()
                single_results.append({
                    "dims": [d], "ops": ["<="], "thresholds": [float(thresh)],
                    "hr": hr, "n": int(below.sum()), "pct": pct,
                    "edge": abs(hr - 0.5)
                })

    single_results.sort(key=lambda x: x["edge"], reverse=True)

    # Step 2: Combine top singles into pairs
    top_singles = single_results[:30]
    pair_results = []

    for i in range(len(top_singles)):
        for j in range(i + 1, min(len(top_singles), 20)):
            s1, s2 = top_singles[i], top_singles[j]
            if s1["dims"][0] == s2["dims"][0]:
                continue  # skip same dim

            # Apply both conditions
            mask1 = np.ones(n, dtype=bool)
            for d, op, t in zip(s1["dims"], s1["ops"], s1["thresholds"]):
                if op == ">":
                    mask1 &= dim_vals[d] > t
                else:
                    mask1 &= dim_vals[d] <= t

            mask2 = np.ones(n, dtype=bool)
            for d, op, t in zip(s2["dims"], s2["ops"], s2["thresholds"]):
                if op == ">":
                    mask2 &= dim_vals[d] > t
                else:
                    mask2 &= dim_vals[d] <= t

            combined = mask1 & mask2 & valid
            cnt = combined.sum()
            if cnt < min_samples:
                continue

            hr = (labels[combined] == 1).mean()
            pair_results.append({
                "dims": s1["dims"] + s2["dims"],
                "ops": s1["ops"] + s2["ops"],
                "thresholds": s1["thresholds"] + s2["thresholds"],
                "hr": hr, "n": cnt,
                "edge": abs(hr - 0.5),
            })

    all_results = single_results + pair_results
    all_results.sort(key=lambda x: x["edge"], reverse=True)

    # Add direction
    for r in all_results:
        r["direction"] = 1 if r["hr"] > 0.5 else -1

    return all_results[:50]


def validate_oos(conditions, states, labels, test_mask, min_samples=15):
    """Validate conditions on out-of-sample data."""
    n = len(states)
    valid = (labels != 0) & test_mask

    dim_vals = {}
    sample = None
    for i in range(n):
        if states[i] is not None:
            sample = states[i]
            break
    if not sample:
        return []

    for d in sample.keys():
        dim_vals[d] = np.array([states[i].get(d, 0) if states[i] else 0 for i in range(n)])

    validated = []
    for cond in conditions:
        mask = valid.copy()
        for d, op, t in zip(cond["dims"], cond["ops"], cond["thresholds"]):
            if d not in dim_vals:
                mask[:] = False
                break
            if op == ">":
                mask &= dim_vals[d] > t
            else:
                mask &= dim_vals[d] <= t

        cnt = mask.sum()
        if cnt < min_samples:
            continue

        hr = (labels[mask] == 1).mean()
        validated.append({
            **cond,
            "oos_hr": hr,
            "oos_n": cnt,
            "oos_edge": abs(hr - 0.5),
            "degradation": cond["hr"] - hr if cond["direction"] == 1 else hr - cond["hr"],
            "oos_correct_direction": (
                (cond["direction"] == 1 and hr > 0.5) or
                (cond["direction"] == -1 and hr < 0.5)
            )
        })

    validated.sort(key=lambda x: x["oos_edge"], reverse=True)
    return validated


def main():
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)

    print(f"Data: {n:,} candles | ${closes.min():,.0f} — ${closes.max():,.0f}")
    print(f"Date: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # Avg price and volatility context
    avg_price = closes.mean()
    returns = np.diff(closes) / closes[:-1]
    avg_1min_move = np.mean(np.abs(np.diff(closes)))
    std_1min = np.std(np.diff(closes))
    atr_20 = pd.Series(np.abs(np.diff(closes))).rolling(20).mean().dropna()

    print(f"Avg price: ${avg_price:,.0f}")
    print(f"Avg 1-min absolute move: ${avg_1min_move:.2f}")
    print(f"Std 1-min move: ${std_1min:.2f}")
    print(f"Avg ATR(20): ${atr_20.mean():.2f}")
    print(f"Typical hourly move (√60 × std): ${std_1min * np.sqrt(60):.2f}")
    print(f"Typical daily move (√1440 × std): ${std_1min * np.sqrt(1440):.2f}")
    print()

    # Build states once
    print("Building state vectors...")
    t0 = time.time()
    states = build_states(closes, volumes)
    valid_count = sum(1 for s in states if s is not None)
    print(f"  Done in {time.time()-t0:.1f}s | {valid_count:,} valid states")
    print()

    # Train/test split
    split = int(n * 0.67)
    train_mask = np.array([states[i] is not None and i < split for i in range(n)])
    test_mask = np.array([states[i] is not None and i >= split for i in range(n)])
    print(f"Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")
    print()

    # ── SWEEP X ──
    # Target values based on volatility structure
    # 1 ATR ≈ $27, so sweep from sub-ATR to multi-ATR
    targets = [
        # (X dollars, max lookahead bars, description)
        (5, 60, "noise"),
        (10, 60, "~0.3 ATR"),
        (15, 90, "~0.5 ATR"),
        (20, 90, "~0.7 ATR"),
        (27, 120, "~1 ATR"),
        (40, 120, "~1.5 ATR"),
        (50, 180, "~2 ATR"),
        (75, 240, "~3 ATR"),
        (100, 360, "~4 ATR"),
        (150, 480, "~6 ATR"),
        (200, 720, "~8 ATR"),
        (300, 1080, "~11 ATR"),
        (500, 1440, "~19 ATR"),
        (750, 2160, "~28 ATR"),
        (1000, 2880, "~37 ATR"),
        (1500, 4320, "~56 ATR"),
        (2000, 5760, "~74 ATR"),
    ]

    print("=" * 100)
    print(f"{'X':>7} {'MaxBars':>8} {'UP':>6} {'DN':>6} {'None':>6} {'Base':>6} | "
          f"{'Best_IS':>8} {'IS_n':>6} {'Best_OOS':>9} {'OOS_n':>6} {'Deg':>6} {'Valid':>5} | Condition")
    print("─" * 100)

    summary = []

    for target, max_bars, desc in targets:
        if max_bars >= n * 0.3:  # skip if lookahead too large relative to data
            print(f"${target:>6} {max_bars:>7}  — skipped (lookahead too large for data)")
            continue

        labels, bars_to_hit = label_moves(closes, target, max_bars)
        n_up = (labels == 1).sum()
        n_down = (labels == -1).sum()
        n_none = (labels == 0).sum()

        if n_up + n_down < 200:
            print(f"${target:>6} {max_bars:>7}  — skipped (only {n_up+n_down} labeled)")
            continue

        base_rate = n_up / (n_up + n_down)

        # Find best conditions on train
        conditions = find_best_conditions(states, labels, train_mask, min_samples=30)

        if not conditions:
            print(f"${target:>6} {max_bars:>7}  {n_up:>5} {n_down:>5} {n_none:>5} {base_rate:>5.1%} | "
                  f"  —       —        —       —      — | No conditions found")
            continue

        best_is = conditions[0]

        # Validate OOS
        validated = validate_oos(conditions[:20], states, labels, test_mask)

        if validated:
            best_oos = validated[0]
            # Use DIRECTIONAL hit rate: if direction==-1, flip
            oos_hr_directional = best_oos["oos_hr"] if best_oos["direction"] == 1 else (1 - best_oos["oos_hr"])
            is_hr_directional = best_oos["hr"] if best_oos["direction"] == 1 else (1 - best_oos["hr"])

            cond_str = " & ".join(
                f"{d}{op}{t:.4f}" for d, op, t in
                zip(best_oos["dims"], best_oos["ops"], best_oos["thresholds"])
            )
            dir_str = "UP" if best_oos["direction"] == 1 else "DN"
            valid_str = "Y" if best_oos["oos_correct_direction"] else "N"
            deg = best_oos["degradation"]

            print(f"${target:>6} {max_bars:>7}  {n_up:>5} {n_down:>5} {n_none:>5} {base_rate:>5.1%} | "
                  f"{is_hr_directional:>7.1%} {best_oos['n']:>5} "
                  f"{oos_hr_directional:>8.1%} {best_oos['oos_n']:>5} "
                  f"{deg:>+5.1%} {valid_str:>5} | {dir_str} when {cond_str:.50s}")

            summary.append({
                "target": target,
                "max_bars": max_bars,
                "desc": desc,
                "base_rate": base_rate,
                "n_up": n_up,
                "n_down": n_down,
                "best_is_hr": is_hr_directional,
                "best_is_n": best_oos["n"],
                "best_oos_hr": oos_hr_directional,
                "best_oos_n": best_oos["oos_n"],
                "degradation": deg,
                "valid": best_oos["oos_correct_direction"],
                "condition": cond_str,
                "direction": dir_str,
                "all_validated": validated,
            })
        else:
            is_hr = best_is["hr"] if best_is["direction"] == 1 else (1 - best_is["hr"])
            print(f"${target:>6} {max_bars:>7}  {n_up:>5} {n_down:>5} {n_none:>5} {base_rate:>5.1%} | "
                  f"{is_hr:>7.1%} {best_is['n']:>5}       —       —      — | (no OOS validation)")

    print("─" * 100)
    print()

    # ── DETAILED ANALYSIS of promising targets ──
    print()
    print("=" * 80)
    print("DETAILED ANALYSIS — Top Validated Conditions per Target")
    print("=" * 80)

    for s in summary:
        if not s["valid"]:
            continue

        print(f"\n{'─' * 70}")
        print(f"TARGET: ${s['target']} ({s['desc']}) | Max {s['max_bars']} bars | Base: {s['base_rate']:.1%}")
        print(f"{'─' * 70}")

        for i, v in enumerate(s["all_validated"][:10]):
            oos_hr = v["oos_hr"] if v["direction"] == 1 else (1 - v["oos_hr"])
            is_hr = v["hr"] if v["direction"] == 1 else (1 - v["hr"])
            dir_str = "UP" if v["direction"] == 1 else "DN"
            cond_str = " & ".join(f"{d}{op}{t:.5f}" for d, op, t in zip(v["dims"], v["ops"], v["thresholds"]))
            valid_str = "OK" if v["oos_correct_direction"] else "FAIL"
            deg = v["degradation"]

            print(f"  #{i+1:2d} {dir_str} IS={is_hr:.1%}(n={v['n']:,}) "
                  f"OOS={oos_hr:.1%}(n={v['oos_n']:,}) "
                  f"deg={deg:+.1%} [{valid_str}] | {cond_str:.55s}")

    # ── FRONTIER ANALYSIS: accuracy vs signal count ──
    print()
    print("=" * 80)
    print("ACCURACY FRONTIER — Max OOS HR at each target (validated conditions only)")
    print("=" * 80)
    print()
    print(f"{'Target':>8} {'OOS HR':>8} {'OOS n':>7} {'IS HR':>8} {'Deg':>7} {'Dir':>4} | Đánh giá")
    print("─" * 70)

    for s in summary:
        if not s["valid"]:
            continue
        eval_str = ""
        hr = s["best_oos_hr"]
        deg = abs(s["degradation"])
        if hr >= 0.90:
            eval_str = "TARGET 91% — CẦN VERIFY THÊM"
        elif hr >= 0.70:
            eval_str = "Có tiềm năng"
        elif hr >= 0.55:
            eval_str = "Edge nhỏ"
        else:
            eval_str = "Không đáng kể"
        if deg > 0.10:
            eval_str += " (OVERFIT risk)"

        print(f"${s['target']:>7} {s['best_oos_hr']:>7.1%} {s['best_oos_n']:>6,} "
              f"{s['best_is_hr']:>7.1%} {s['degradation']:>+6.1%} {s['direction']:>4} | {eval_str}")

    print("─" * 70)
    print()

    # ── CONCLUSION ──
    if summary:
        # Find the highest validated OOS HR
        valid_summary = [s for s in summary if s["valid"]]
        if valid_summary:
            best = max(valid_summary, key=lambda x: x["best_oos_hr"])
            print(f"BEST VALIDATED: ${best['target']} → {best['best_oos_hr']:.1%} OOS hit rate "
                  f"({best['best_oos_n']:,} signals)")
            print(f"  Condition: {best['direction']} when {best['condition']}")
            print(f"  IS: {best['best_is_hr']:.1%} | Degradation: {best['degradation']:+.1%}")
            print()

        # Is 91% achievable?
        achievable = [s for s in valid_summary if s["best_oos_hr"] >= 0.91]
        near = [s for s in valid_summary if 0.80 <= s["best_oos_hr"] < 0.91]

        if achievable:
            print("91% ACHIEVED at:")
            for s in achievable:
                print(f"  ${s['target']} → {s['best_oos_hr']:.1%} ({s['best_oos_n']} signals)")
        elif near:
            print("91% NOT yet achieved, but CLOSE:")
            for s in near:
                print(f"  ${s['target']} → {s['best_oos_hr']:.1%} ({s['best_oos_n']} signals)")
            print("\nĐể đạt 91%: cần thêm data HOẶC thêm điều kiện lọc (giảm signals, tăng precision)")
        else:
            max_hr = max(s["best_oos_hr"] for s in valid_summary) if valid_summary else 0
            print(f"91% chưa đạt được. Max OOS: {max_hr:.1%}")
            print("Cần: nhiều data hơn + feature engineering tốt hơn")


if __name__ == "__main__":
    main()
