#!/usr/bin/env python3
"""
Bài Toán Mới: Tìm P mà giá sẽ chạm CẢ P+X và P-X
=====================================================

Không đoán hướng. Đoán BIÊN ĐỘ.

Tại điểm P, trong N bars tiếp theo:
  - Giá có chạm P+X không?     (yes/no)
  - Giá có chạm P-X không?     (yes/no)

Outcomes:
  BOTH  = chạm cả 2 → Straddle thắng
  UP    = chỉ chạm P+X
  DOWN  = chỉ chạm P-X
  NONE  = không chạm bên nào

Strategy nếu predict BOTH đúng:
  - Đặt BUY limit + SELL limit cùng lúc
  - Cả 2 sẽ TP → profit = 2X - spread - fees
  - Hoặc: BUY entry, TP +X, đồng thời SELL entry, TP -X
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
    metric_history = []
    states = [None] * n
    dims = ["vol", "autocorr", "skew", "kurt", "vol_anom", "pv_corr"]
    for i in range(110, n):
        r_w = returns[max(0, i-200):i+1]
        v_w = volumes[max(0, i-200):i+1]
        m = compute_metrics(r_w, v_w)
        metric_history.append(m)
        if len(metric_history) < 3: continue
        cur, prev, prev2 = metric_history[-1], metric_history[-2], metric_history[-3]
        state = {}
        for nm in dims:
            state[nm] = cur[nm]
            state[f"d_{nm}"] = cur[nm] - prev[nm]
            state[f"dd_{nm}"] = (cur[nm] - prev[nm]) - (prev[nm] - prev2[nm])
        if len(metric_history) >= 12:
            vels = []
            for j in range(max(0, len(metric_history)-20), len(metric_history)-1):
                vels.append({nm: metric_history[j+1][nm] - metric_history[j][nm] for nm in dims})
            if len(vels) >= 5:
                for ii, n1 in enumerate(dims):
                    for n2 in dims[ii+1:]:
                        v1 = np.array([v[n1] for v in vels])
                        v2 = np.array([v[n2] for v in vels])
                        c = np.corrcoef(v1, v2)[0,1] if np.std(v1)>1e-12 and np.std(v2)>1e-12 else 0
                        state[f"xc_{n1}_{n2}"] = 0 if np.isnan(c) else float(c)
            else:
                for ii, n1 in enumerate(dims):
                    for n2 in dims[ii+1:]:
                        state[f"xc_{n1}_{n2}"] = 0
        else:
            for ii, n1 in enumerate(dims):
                for n2 in dims[ii+1:]:
                    state[f"xc_{n1}_{n2}"] = 0
        states[i] = state
    return states


def label_both_sides(closes, target_x, max_bars):
    """
    Label each bar:
      BOTH=3: price hits both +X and -X within max_bars
      UP=1:   price hits only +X
      DOWN=2: price hits only -X
      NONE=0: neither reached
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    time_to_both = np.zeros(n, dtype=np.int32)  # bars until BOTH hit

    for i in range(n - 1):
        base = closes[i]
        up_t = base + target_x
        dn_t = base - target_x
        end = min(i + max_bars + 1, n)

        hit_up = hit_dn = False
        time_up = time_dn = -1

        for j in range(i + 1, end):
            if not hit_up and closes[j] >= up_t:
                hit_up = True
                time_up = j - i
            if not hit_dn and closes[j] <= dn_t:
                hit_dn = True
                time_dn = j - i
            if hit_up and hit_dn:
                break

        if hit_up and hit_dn:
            labels[i] = 3  # BOTH
            time_to_both[i] = max(time_up, time_dn)
        elif hit_up:
            labels[i] = 1  # UP only
        elif hit_dn:
            labels[i] = 2  # DOWN only
        else:
            labels[i] = 0  # NONE

    return labels, time_to_both


def walk_forward_both(closes, states, target_x, max_bars, condition_func, n_folds=5):
    """Walk-forward test for BOTH-sides prediction."""
    n = len(closes)
    labels, times = label_both_sides(closes, target_x, max_bars)

    valid_indices = [i for i in range(n) if states[i] is not None and labels[i] != 0]
    if len(valid_indices) < n_folds * 5:
        return None

    fold_size = len(valid_indices) // n_folds
    fold_results = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = (fold+1) * fold_size if fold < n_folds-1 else len(valid_indices)
        test_idx = valid_indices[start:end]

        matches = [(i, labels[i], times[i]) for i in test_idx if condition_func(states[i])]
        if len(matches) < 2:
            fold_results.append({"n": 0, "both": 0, "up": 0, "dn": 0})
            continue

        both = sum(1 for _, l, _ in matches if l == 3)
        up = sum(1 for _, l, _ in matches if l == 1)
        dn = sum(1 for _, l, _ in matches if l == 2)
        avg_time = np.mean([t for _, l, t in matches if l == 3 and t > 0]) if both > 0 else 0

        fold_results.append({
            "n": len(matches), "both": both, "up": up, "dn": dn,
            "both_rate": both / len(matches) if len(matches) > 0 else 0,
            "avg_time_to_both": avg_time,
        })

    total_n = sum(f["n"] for f in fold_results)
    total_both = sum(f["both"] for f in fold_results)
    if total_n == 0:
        return None

    both_rate = total_both / total_n
    fold_rates = [f["both_rate"] for f in fold_results if f["n"] > 0]

    try:
        # Compare to base rate
        all_labeled = labels[labels != 0]
        base_both = (all_labeled == 3).mean()
        p = stats.binomtest(total_both, total_n, base_both).pvalue if base_both > 0 else 1.0
    except:
        p = 1.0

    return {
        "total_n": total_n, "total_both": total_both,
        "both_rate": both_rate,
        "base_both_rate": float((labels[labels != 0] == 3).mean()),
        "p_value": p,
        "fold_rates": fold_rates,
        "fold_details": fold_results,
        "consistent": sum(1 for r in fold_rates if r > 0.5),
        "active": len(fold_rates),
    }


def main():
    # Try full data first, fallback to regular
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min_full.parquet"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"

    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)

    avg_move = np.mean(np.abs(np.diff(closes)))
    print(f"Data: {n:,} candles | ${closes.min():,.0f} — ${closes.max():,.0f}")
    print(f"1 ATR ≈ ${avg_move:.0f}")
    print()

    print("Building states...")
    t0 = time.time()
    states = build_states(closes, volumes)
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    # ── Base rates: how often does BOTH hit? ──
    print("=" * 80)
    print("BASE RATES: How often does price hit BOTH +$X and -$X?")
    print("=" * 80)
    print(f"{'X':>7} {'MaxB':>6} {'BOTH':>7} {'UP':>7} {'DN':>7} {'NONE':>7} {'BOTH%':>7} {'AvgTime':>8}")
    print("─" * 65)

    atr = avg_move
    for mult in [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20]:
        x = round(atr * mult)
        if x < 1: continue
        mb = max(60, int(mult * 15))
        if mb > n * 0.2: continue

        labels, times = label_both_sides(closes, x, mb)
        both = (labels == 3).sum()
        up = (labels == 1).sum()
        dn = (labels == 2).sum()
        none = (labels == 0).sum()
        both_pct = both / max(both + up + dn, 1)
        avg_t = np.mean(times[labels == 3]) if both > 0 else 0

        print(f"${x:>6} {mb:>5} {both:>6,} {up:>6,} {dn:>6,} {none:>6,} {both_pct:>6.1%} {avg_t:>7.0f}m")

    # ── Adaptive conditions ──
    print()
    print("=" * 80)
    print("CONDITION SEARCH: When is BOTH more likely?")
    print("=" * 80)

    vol_vals = [s["vol"] for s in states if s is not None]
    kurt_vals = [s["kurt"] for s in states if s is not None]
    ac_vals = [s["autocorr"] for s in states if s is not None]
    va_vals = [s["vol_anom"] for s in states if s is not None]
    pv_vals = [s["pv_corr"] for s in states if s is not None]

    vol_p80 = np.percentile(vol_vals, 80)
    vol_p90 = np.percentile(vol_vals, 90)
    vol_p10 = np.percentile(vol_vals, 10)
    kurt_p80 = np.percentile(kurt_vals, 80)
    kurt_p90 = np.percentile(kurt_vals, 90)
    va_p80 = np.percentile(va_vals, 80)
    va_p90 = np.percentile(va_vals, 90)
    pv_p80 = np.percentile(pv_vals, 80)

    print(f"  Thresholds: vol_p90={vol_p90:.6f} kurt_p80={kurt_p80:.2f} vol_anom_p80={va_p80:.2f}")

    conditions = {
        # Hypothesis: High vol = range expanding → BOTH more likely
        "HIGH_VOL": lambda s: s.get("vol", 0) >= vol_p80,
        "VERY_HIGH_VOL": lambda s: s.get("vol", 0) >= vol_p90,

        # Hypothesis: High kurtosis = fat tails → extreme moves → BOTH
        "HIGH_KURT": lambda s: s.get("kurt", 0) >= kurt_p80,
        "VERY_HIGH_KURT": lambda s: s.get("kurt", 0) >= kurt_p90,

        # Hypothesis: Vol + Kurt combo = chaos → BOTH
        "VOL_KURT_HIGH": lambda s: s.get("vol", 0) >= vol_p80 and s.get("kurt", 0) >= kurt_p80,

        # Hypothesis: Volume spike = something happening → BOTH
        "VOL_SPIKE": lambda s: s.get("vol_anom", 0) >= va_p80,
        "BIG_VOL_SPIKE": lambda s: s.get("vol_anom", 0) >= va_p90,

        # Hypothesis: Low vol → QUIET → NOT both (control)
        "LOW_VOL": lambda s: s.get("vol", 0) <= vol_p10,

        # Hypothesis: Vol increasing (d_vol > 0) + high vol = breakout
        "VOL_ACCEL": lambda s: s.get("d_vol", 0) > 0 and s.get("vol", 0) >= vol_p80,

        # Hypothesis: High PV correlation = volume confirms moves → range expanding
        "HIGH_PV": lambda s: s.get("pv_corr", 0) >= pv_p80 and s.get("vol", 0) >= vol_p80,

        # Hypothesis: Low autocorr + high vol = chaotic, no direction, wide range
        "CHAOS_MODE": lambda s: abs(s.get("autocorr", 0)) < 0.05 and s.get("vol", 0) >= vol_p80 and s.get("kurt", 0) >= kurt_p80,
    }

    # Test target sweep × conditions
    target_mults = [1, 1.5, 2, 3, 5, 7, 10]

    for cond_name, cond_func in conditions.items():
        trigger_count = sum(1 for s in states if s is not None and cond_func(s))
        if trigger_count < 20:
            continue

        best_both_rate = 0
        best_x = 0
        best_result = None

        for mult in target_mults:
            x = round(atr * mult)
            mb = max(60, int(mult * 15))
            if mb > n * 0.2: continue

            result = walk_forward_both(closes, states, x, mb, cond_func, n_folds=5)
            if result and result["total_n"] >= 10 and result["both_rate"] > best_both_rate:
                best_both_rate = result["both_rate"]
                best_x = x
                best_result = result

        if best_result and best_result["total_n"] >= 10:
            base = best_result["base_both_rate"]
            lift = best_both_rate - base
            sig = "***" if best_result["p_value"] < 0.001 else "**" if best_result["p_value"] < 0.01 else "*" if best_result["p_value"] < 0.05 else ""
            fold_str = " ".join(f"{r:.0%}" for r in best_result["fold_rates"])
            print(f"  {cond_name:20s} X=${best_x:>5} BOTH={best_both_rate:.1%} (base={base:.1%} lift={lift:+.1%}) "
                  f"n={best_result['total_n']:>4} p={best_result['p_value']:.4f}{sig} "
                  f"[{fold_str}] trig={trigger_count:,}")

    # ── Detailed analysis for best conditions ──
    print()
    print("=" * 80)
    print("DETAILED: X sweep for top conditions")
    print("=" * 80)

    top_conditions = ["HIGH_VOL", "VOL_KURT_HIGH", "VOL_ACCEL", "CHAOS_MODE", "LOW_VOL"]

    for cond_name in top_conditions:
        if cond_name not in conditions:
            continue
        cond_func = conditions[cond_name]
        trigger_count = sum(1 for s in states if s is not None and cond_func(s))
        if trigger_count < 10:
            continue

        print(f"\n  {cond_name} ({trigger_count:,} triggers):")
        print(f"  {'X':>7} {'BOTH%':>7} {'Base%':>7} {'Lift':>7} {'N':>5} {'p':>8} | Folds")
        print(f"  {'─'*65}")

        for mult in [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15]:
            x = round(atr * mult)
            mb = max(60, int(mult * 15))
            if mb > n * 0.2: continue

            result = walk_forward_both(closes, states, x, mb, cond_func, n_folds=5)
            if not result or result["total_n"] < 5:
                continue

            base = result["base_both_rate"]
            lift = result["both_rate"] - base
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
            fold_str = " ".join(f"{r:.0%}" for r in result["fold_rates"])

            marker = ""
            if result["both_rate"] >= 0.90:
                marker = " ★★★"
            elif result["both_rate"] >= 0.80:
                marker = " ★★"
            elif result["both_rate"] >= 0.70:
                marker = " ★"

            print(f"  ${x:>6} {result['both_rate']:>6.1%} {base:>6.1%} {lift:>+6.1%} "
                  f"{result['total_n']:>5} {result['p_value']:>7.4f}{sig:>1} | [{fold_str}]{marker}")

    # ── STRATEGY SIMULATION ──
    print()
    print("=" * 80)
    print("STRATEGY: Straddle P/L if we trade BOTH predictions")
    print("=" * 80)
    print()
    print("Strategy: When condition triggers, open BOTH directions")
    print("  BUY  entry → TP at +$X")
    print("  SELL entry → TP at -$X")
    print("  Max hold = max_bars, then close whatever is open")
    print()

    # Simulate for VOL_KURT_HIGH at a few targets
    for cond_name in ["VOL_KURT_HIGH", "HIGH_VOL", "VOL_ACCEL"]:
        if cond_name not in conditions:
            continue
        cond_func = conditions[cond_name]

        for mult in [1.5, 2, 3, 5]:
            x = round(atr * mult)
            mb = max(60, int(mult * 15))
            if mb > n * 0.2: continue

            labels, times = label_both_sides(closes, x, mb)
            pnl_trades = []

            for i in range(n):
                if states[i] is None: continue
                if not cond_func(states[i]): continue
                if i + mb >= n: continue

                future = closes[i+1:i+mb+1] - closes[i]
                if len(future) == 0: continue

                # BUY side: TP at +X
                buy_pnl = 0
                for p in future:
                    if p >= x:
                        buy_pnl = x
                        break
                else:
                    buy_pnl = float(future[-1])  # close at end

                # SELL side: TP at -X
                sell_pnl = 0
                for p in future:
                    if p <= -x:
                        sell_pnl = x  # profit (short TP)
                        break
                else:
                    sell_pnl = float(-future[-1])  # close at end

                total_pnl = buy_pnl + sell_pnl
                pnl_trades.append(total_pnl)

            if not pnl_trades:
                continue

            pnl = np.array(pnl_trades)
            win = (pnl > 0).sum()
            total = len(pnl)
            avg_pnl = pnl.mean()
            total_pnl = pnl.sum()
            max_win = pnl.max()
            max_loss = pnl.min()

            print(f"  {cond_name:20s} X=${x:>5} | {total:>4} trades | "
                  f"Win={win/total:.1%} | Avg=${avg_pnl:>+.1f} | "
                  f"Total=${total_pnl:>+,.0f} | Max W=${max_win:>+.0f} L=${max_loss:>+.0f}")


if __name__ == "__main__":
    main()
