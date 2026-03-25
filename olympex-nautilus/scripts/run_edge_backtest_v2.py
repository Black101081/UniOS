#!/usr/bin/env python3
"""
Edge Discovery Backtest V2 — Multi-Target A/B/C/D Test
=======================================================
V2 improvements:
- Test multiple target moves ($3, $50, $100, $200, $500)
- Fix Approach A: lower thresholds, better scanning
- Add percentile-based normalization for metrics
- Better regime membership calibration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations, product
from scipy import stats
import warnings
import time
warnings.filterwarnings("ignore")


# ============================================================
# MEASUREMENT ENGINE
# ============================================================

SCALES = [5, 10, 20, 50, 100]


def realized_vol(returns, scale):
    if len(returns) < scale:
        return np.nan
    return float(np.std(returns[-scale:]))


def autocorrelation(returns, scale, lag=1):
    if len(returns) < scale + lag:
        return np.nan
    r = returns[-scale:]
    if np.std(r) < 1e-12:
        return 0.0
    return float(np.corrcoef(r[lag:], r[:-lag])[0, 1])


def skewness_metric(returns, scale):
    if len(returns) < scale:
        return np.nan
    r = returns[-scale:]
    s = np.std(r)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((r - np.mean(r)) / s) ** 3))


def kurtosis_metric(returns, scale):
    if len(returns) < scale:
        return np.nan
    r = returns[-scale:]
    s = np.std(r)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((r - np.mean(r)) / s) ** 4))


def volume_anomaly(volumes, scale):
    if len(volumes) < scale:
        return np.nan
    window = volumes[-scale:]
    med = np.median(window)
    if med < 1e-12:
        return 1.0
    return float(volumes[-1] / med)


def pv_correlation(returns, volumes, scale):
    if len(returns) < scale or len(volumes) < scale:
        return np.nan
    r = np.abs(returns[-scale:])
    v = volumes[-scale:]
    if np.std(r) < 1e-12 or np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(r, v)[0, 1])


def compute_base_metrics(returns, volumes):
    result = {}
    for name, func in [
        ("vol", lambda r, v, s: realized_vol(r, s)),
        ("autocorr", lambda r, v, s: autocorrelation(r, s)),
        ("skew", lambda r, v, s: skewness_metric(r, s)),
        ("kurt", lambda r, v, s: kurtosis_metric(r, s)),
        ("vol_anom", lambda r, v, s: volume_anomaly(v, s)),
        ("pv_corr", lambda r, v, s: pv_correlation(r, v, s)),
    ]:
        values = [func(returns, volumes, s) for s in SCALES if len(returns) >= s]
        values = [v for v in values if not np.isnan(v)]
        result[name] = float(np.mean(values)) if values else 0.0
    return result


def compute_state_vector(metric_history):
    if len(metric_history) < 3:
        return None
    current = metric_history[-1]
    prev = metric_history[-2]
    prev2 = metric_history[-3]
    dim_names = ["vol", "autocorr", "skew", "kurt", "vol_anom", "pv_corr"]
    state = {}

    for name in dim_names:
        state[name] = current[name]
    for name in dim_names:
        state[f"d_{name}"] = current[name] - prev[name]
    for name in dim_names:
        v_now = current[name] - prev[name]
        v_prev = prev[name] - prev2[name]
        state[f"dd_{name}"] = v_now - v_prev

    if len(metric_history) >= 12:
        vel_history = []
        for i in range(max(0, len(metric_history) - 20), len(metric_history) - 1):
            vel = {n: metric_history[i + 1][n] - metric_history[i][n] for n in dim_names}
            vel_history.append(vel)
        if len(vel_history) >= 5:
            for i, n1 in enumerate(dim_names):
                for n2 in dim_names[i + 1:]:
                    v1 = np.array([v[n1] for v in vel_history])
                    v2 = np.array([v[n2] for v in vel_history])
                    if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
                        state[f"xc_{n1}_{n2}"] = 0.0
                    else:
                        c = np.corrcoef(v1, v2)[0, 1]
                        state[f"xc_{n1}_{n2}"] = 0.0 if np.isnan(c) else float(c)
        else:
            for i, n1 in enumerate(dim_names):
                for n2 in dim_names[i + 1:]:
                    state[f"xc_{n1}_{n2}"] = 0.0
    else:
        for i, n1 in enumerate(dim_names):
            for n2 in dim_names[i + 1:]:
                state[f"xc_{n1}_{n2}"] = 0.0

    # Additional derived features: recent return direction & magnitude
    state["ret_mean_5"] = float(np.mean(list(metric_history[i]["skew"] for i in range(-min(5, len(metric_history)), 0))))
    state["vol_trend"] = state["d_vol"]  # alias for clarity

    return state


# ============================================================
# REGIME MEMBERSHIP
# ============================================================

def compute_regime_membership(state):
    # Normalize to 0-1 using empirical knowledge
    vol_raw = state.get("vol", 0)
    v = min(vol_raw / 0.005, 1.0) if vol_raw > 0 else 0.0  # 0.5% vol = max
    a = np.clip(state.get("autocorr", 0), -1, 1)
    k_raw = state.get("kurt", 3)
    k = min(max((k_raw - 2) / 8, 0), 1)
    va_raw = state.get("vol_anom", 1)
    va = min(max((va_raw - 0.5) / 4, 0), 1)

    quiet = max(0.001, (1 - v) * (1 - abs(a)) * (1 - k))
    momentum = max(0.001, max(a, 0) * (1 - k) * (0.3 + va * 0.7))
    revert = max(0.001, max(-a, 0) * (1 - k) * (1 - va * 0.5))
    chaos = max(0.001, v * k)

    total = quiet + momentum + revert + chaos
    return {
        "QUIET": quiet / total,
        "MOMENTUM": momentum / total,
        "REVERT": revert / total,
        "CHAOS": chaos / total,
    }


# ============================================================
# LABELING
# ============================================================

def label_future_moves(closes, target_move, max_bars=180):
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        base = closes[i]
        up_target = base + target_move
        down_target = base - target_move
        end = min(i + max_bars + 1, n)

        hit_up = -1
        hit_down = -1
        for j in range(i + 1, end):
            if hit_up < 0 and closes[j] >= up_target:
                hit_up = j - i
            if hit_down < 0 and closes[j] <= down_target:
                hit_down = j - i
            if hit_up >= 0 and hit_down >= 0:
                break

        if hit_up >= 0 and hit_down >= 0:
            labels[i] = 1 if hit_up < hit_down else -1
        elif hit_up >= 0:
            labels[i] = 1
        elif hit_down >= 0:
            labels[i] = -1
    return labels


# ============================================================
# APPROACH A: Data-Driven
# ============================================================

def approach_a_scan(states, labels, train_mask):
    valid = (labels != 0) & train_mask
    if valid.sum() < 30:
        return {"edges": [], "dims_ranked": []}

    # Get all dimension names
    sample_state = None
    for i in range(len(states)):
        if states[i] is not None and valid[i]:
            sample_state = states[i]
            break
    if sample_state is None:
        return {"edges": [], "dims_ranked": []}

    dim_names = list(sample_state.keys())
    dim_scores = []

    for dim in dim_names:
        vals = np.array([states[i].get(dim, 0) if states[i] else 0 for i in range(len(states))])
        dim_vals = vals[valid]
        dim_labels = labels[valid]

        if len(dim_vals) < 30 or np.std(dim_vals) < 1e-15:
            continue

        try:
            quintiles = np.percentile(dim_vals, [20, 40, 60, 80])
            if len(np.unique(quintiles)) < 3:
                continue
            bins = np.digitize(dim_vals, quintiles)
        except Exception:
            continue

        hit_rates = []
        counts = []
        for b in range(5):
            mask = bins == b
            cnt = mask.sum()
            if cnt < 5:
                hit_rates.append(0.5)
                counts.append(0)
                continue
            hr = (dim_labels[mask] == 1).mean()
            hit_rates.append(hr)
            counts.append(cnt)

        valid_bins = [i for i in range(5) if counts[i] >= 5]
        if len(valid_bins) < 3:
            continue

        valid_hrs = [hit_rates[i] for i in valid_bins]
        rho, _ = stats.spearmanr(range(len(valid_hrs)), valid_hrs)
        spread = max(valid_hrs) - min(valid_hrs)

        if not np.isnan(rho):
            dim_scores.append({
                "dim": dim,
                "monotonicity": rho,
                "spread": spread,
                "hit_rates": hit_rates,
                "counts": counts,
                "score": abs(rho) * spread,
            })

    dim_scores.sort(key=lambda x: x["score"], reverse=True)
    top_dims = [d["dim"] for d in dim_scores[:8]]

    # Find edges - single dims and pairs
    edges = []
    for combo_size in [1, 2]:
        for combo in combinations(top_dims[:6], combo_size):
            vals_list = [np.array([states[i].get(d, 0) if states[i] else 0 for i in range(len(states))]) for d in combo]

            # Use tertiles: LOW (bottom 33%), MID, HIGH (top 33%)
            percentiles = [np.percentile(v[valid], [33, 67]) for v in vals_list]

            configs = [("LOW", "HIGH")] if combo_size == 1 else list(product(["LOW", "HIGH"], repeat=combo_size))

            for config in configs:
                mask = valid.copy()
                for ci in range(combo_size):
                    if config[ci] == "HIGH":
                        mask &= (vals_list[ci] > percentiles[ci][1])
                    else:
                        mask &= (vals_list[ci] < percentiles[ci][0])

                count = mask.sum()
                if count < 15:
                    continue

                hr = (labels[mask] == 1).mean()
                if abs(hr - 0.5) < 0.015:  # lowered threshold
                    continue

                k = int((labels[mask] == 1).sum())
                try:
                    p_val = stats.binomtest(k, count, 0.5).pvalue
                except Exception:
                    continue

                thresholds = []
                for ci in range(combo_size):
                    if config[ci] == "HIGH":
                        thresholds.append(float(percentiles[ci][1]))
                    else:
                        thresholds.append(float(percentiles[ci][0]))

                edges.append({
                    "dims": list(combo),
                    "config": list(config),
                    "thresholds": thresholds,
                    "hit_rate": float(hr),
                    "count": int(count),
                    "p_value": float(p_val),
                    "direction": 1 if hr > 0.5 else -1,
                    "edge_strength": abs(hr - 0.5),
                })

    edges.sort(key=lambda x: x["edge_strength"] * min(x["count"], 100), reverse=True)
    return {"edges": edges[:30], "dims_ranked": dim_scores[:10]}


def approach_a_predict(state, edges):
    if not edges:
        return 0, 0.0

    votes_up = 0.0
    votes_down = 0.0
    n_match = 0

    for edge in edges:
        match = True
        for i, dim in enumerate(edge["dims"]):
            val = state.get(dim, 0)
            if edge["config"][i] == "HIGH" and val <= edge["thresholds"][i]:
                match = False
                break
            if edge["config"][i] == "LOW" and val >= edge["thresholds"][i]:
                match = False
                break

        if match:
            weight = edge["edge_strength"] * min(edge["count"], 200) / 200
            if edge["direction"] == 1:
                votes_up += weight
            else:
                votes_down += weight
            n_match += 1

    if n_match == 0:
        return 0, 0.0

    total = votes_up + votes_down
    if total < 0.001:
        return 0, 0.0

    if votes_up > votes_down:
        return 1, votes_up / total
    else:
        return -1, votes_down / total


# ============================================================
# APPROACH B: Theory-Driven
# ============================================================

def approach_b_predict(state, membership):
    signals = []
    autocorr = state.get("autocorr", 0)
    sk = state.get("skew", 0)
    d_autocorr = state.get("d_autocorr", 0)
    d_vol = state.get("d_vol", 0)
    d_vol_anom = state.get("d_vol_anom", 0)
    pv = state.get("pv_corr", 0)

    # B1: Momentum continuation
    if membership["MOMENTUM"] > 0.35 and autocorr > 0.03:
        strength = membership["MOMENTUM"] * abs(autocorr)
        if sk > 0.1:
            signals.append((1, strength))
        elif sk < -0.1:
            signals.append((-1, strength))

    # B2: Mean reversion
    if membership["REVERT"] > 0.35 and autocorr < -0.03:
        strength = membership["REVERT"] * abs(autocorr)
        if sk > 0.3:
            signals.append((-1, strength * 0.7))
        elif sk < -0.3:
            signals.append((1, strength * 0.7))

    # B3: Breakout from quiet
    if membership["QUIET"] > 0.3 and d_autocorr > 0.01 and d_vol > 0:
        strength = abs(d_autocorr) * 3
        if sk > 0:
            signals.append((1, strength))
        elif sk < 0:
            signals.append((-1, strength))

    # B4: CHAOS filter
    if membership["CHAOS"] > 0.45:
        return 0, 0.0

    # B5: Volume-momentum divergence
    if d_vol_anom > 0.05 and d_autocorr < -0.01:
        if sk > 0:
            signals.append((-1, 0.2))
        elif sk < 0:
            signals.append((1, 0.2))

    # B6: Strong PV confirmation — new
    if pv > 0.3 and abs(autocorr) > 0.05:
        strength = pv * abs(autocorr)
        direction = 1 if sk > 0 else -1 if sk < 0 else 0
        if direction != 0:
            signals.append((direction, strength))

    if not signals:
        return 0, 0.0

    total_up = sum(s[1] for s in signals if s[0] == 1)
    total_down = sum(s[1] for s in signals if s[0] == -1)

    if total_up > total_down and total_up > 0.02:
        return 1, min(total_up, 1.0)
    elif total_down > total_up and total_down > 0.02:
        return -1, min(total_down, 1.0)
    return 0, 0.0


# ============================================================
# APPROACH C: Combined
# ============================================================

def approach_c_predict(state, membership, edges):
    if membership.get("CHAOS", 0) > 0.45:
        return 0, 0.0

    b_dir, b_conf = approach_b_predict(state, membership)
    a_dir, a_conf = approach_a_predict(state, edges)

    if a_dir != 0 and b_dir != 0 and a_dir == b_dir:
        return a_dir, (a_conf + b_conf) / 2  # Agreement → strong
    if a_dir != 0 and b_dir == 0:
        return a_dir, a_conf * 0.6
    if b_dir != 0 and a_dir == 0:
        return b_dir, b_conf * 0.6
    if a_dir != 0 and b_dir != 0 and a_dir != b_dir:
        return 0, 0.0  # Disagree → skip
    return 0, 0.0


# ============================================================
# APPROACH D: Random
# ============================================================

def approach_d_predict(rng):
    r = rng.random()
    if r < 0.33:
        return 1, 0.5
    elif r < 0.66:
        return -1, 0.5
    return 0, 0.0


# ============================================================
# BACKTEST
# ============================================================

@dataclass
class Result:
    name: str
    total: int = 0
    correct: int = 0
    wrong: int = 0
    no_signal: int = 0
    skipped: int = 0
    long_n: int = 0
    long_ok: int = 0
    short_n: int = 0
    short_ok: int = 0
    hiconf_n: int = 0
    hiconf_ok: int = 0

    @property
    def hr(self):
        return self.correct / self.total if self.total else 0

    @property
    def long_hr(self):
        return self.long_ok / self.long_n if self.long_n else 0

    @property
    def short_hr(self):
        return self.short_ok / self.short_n if self.short_n else 0

    @property
    def hiconf_hr(self):
        return self.hiconf_ok / self.hiconf_n if self.hiconf_n else 0


def record(res, direction, confidence, actual):
    if direction == 0:
        res.no_signal += 1
        return
    res.total += 1
    ok = (direction == actual)
    if ok:
        res.correct += 1
    else:
        res.wrong += 1
    if direction == 1:
        res.long_n += 1
        if ok:
            res.long_ok += 1
    else:
        res.short_n += 1
        if ok:
            res.short_ok += 1
    if confidence > 0.6:
        res.hiconf_n += 1
        if ok:
            res.hiconf_ok += 1


def run_single_target(closes, volumes, returns, states, target_move, max_bars):
    """Run A/B/C/D test for a single target move size."""
    n = len(closes)

    labels = label_future_moves(closes, target_move, max_bars)
    n_up = (labels == 1).sum()
    n_down = (labels == -1).sum()
    n_none = (labels == 0).sum()

    if n_up + n_down < 100:
        return None

    base_rate = n_up / (n_up + n_down)

    # Train/test split
    split = int(n * 0.67)
    train_mask = np.array([i < split and states[i] is not None for i in range(n)])
    test_mask = np.array([i >= split and states[i] is not None for i in range(n)])

    # Train Approach A
    a_result = approach_a_scan(states, labels, train_mask)

    # Test all approaches
    results = {k: Result(k) for k in ["A", "B", "C", "D"]}
    rng = np.random.default_rng(42)

    for i in np.where(test_mask)[0]:
        if labels[i] == 0:
            for r in results.values():
                r.skipped += 1
            continue

        state = states[i]
        actual = labels[i]
        mem = compute_regime_membership(state)

        a_d, a_c = approach_a_predict(state, a_result["edges"])
        record(results["A"], a_d, a_c, actual)

        b_d, b_c = approach_b_predict(state, mem)
        record(results["B"], b_d, b_c, actual)

        c_d, c_c = approach_c_predict(state, mem, a_result["edges"])
        record(results["C"], c_d, c_c, actual)

        d_d, d_c = approach_d_predict(rng)
        record(results["D"], d_d, d_c, actual)

    # Also get in-sample for A & B
    is_results = {k: Result(k) for k in ["A", "B"]}
    for i in np.where(train_mask)[0]:
        if labels[i] == 0:
            continue
        state = states[i]
        actual = labels[i]
        mem = compute_regime_membership(state)
        a_d, a_c = approach_a_predict(state, a_result["edges"])
        record(is_results["A"], a_d, a_c, actual)
        b_d, b_c = approach_b_predict(state, mem)
        record(is_results["B"], b_d, b_c, actual)

    return {
        "target": target_move,
        "max_bars": max_bars,
        "n_up": n_up,
        "n_down": n_down,
        "n_none": n_none,
        "base_rate": base_rate,
        "results": results,
        "is_results": is_results,
        "n_edges": len(a_result["edges"]),
        "top_dims": a_result["dims_ranked"][:5],
        "top_edges": a_result["edges"][:5],
    }


def main():
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)

    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)
    returns = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])

    print(f"Data: {n:,} candles | ${closes.min():,.0f} — ${closes.max():,.0f}")
    print(f"Date: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print()

    # Compute states once
    print("Computing state vectors...")
    t0 = time.time()
    WARMUP = 110
    metric_history = []
    states = [None] * n
    for i in range(WARMUP, n):
        r_w = returns[max(0, i - 200): i + 1]
        v_w = volumes[max(0, i - 200): i + 1]
        metrics = compute_base_metrics(r_w, v_w)
        metric_history.append(metrics)
        state = compute_state_vector(metric_history)
        if state is not None:
            states[i] = state
    print(f"  Done in {time.time() - t0:.1f}s | {sum(1 for s in states if s):,} valid states")
    print()

    # Test multiple targets
    targets = [
        (3, 120),     # $3 in 2h
        (50, 180),    # $50 in 3h
        (100, 360),   # $100 in 6h
        (200, 720),   # $200 in 12h
        (500, 1440),  # $500 in 24h
    ]

    all_results = []
    for target_move, max_bars in targets:
        print(f"{'=' * 70}")
        print(f"TARGET: ${target_move} move | Max lookahead: {max_bars} bars ({max_bars / 60:.0f}h)")
        print(f"{'=' * 70}")

        t0 = time.time()
        result = run_single_target(closes, volumes, returns, states, target_move, max_bars)
        elapsed = time.time() - t0

        if result is None:
            print(f"  Skipped — not enough labeled data")
            print()
            continue

        all_results.append(result)

        # Labels
        print(f"  Labels: UP={result['n_up']:,} ({result['n_up']/(result['n_up']+result['n_down'])*100:.1f}%) "
              f"DOWN={result['n_down']:,} ({result['n_down']/(result['n_up']+result['n_down'])*100:.1f}%) "
              f"NEITHER={result['n_none']:,} | Base rate: {result['base_rate']:.1%}")
        print(f"  Edges found: {result['n_edges']}")
        print(f"  Computed in {elapsed:.1f}s")

        # Top dims
        if result["top_dims"]:
            print(f"\n  Top predictive dimensions:")
            for d in result["top_dims"][:3]:
                hr_str = " ".join(f"{h:.0%}" for h in d["hit_rates"])
                print(f"    {d['dim']:25s} mono={d['monotonicity']:+.2f} spread={d['spread']:.3f} [{hr_str}]")

        # Top edges
        if result["top_edges"]:
            print(f"\n  Top edges:")
            for e in result["top_edges"][:3]:
                dims_str = " & ".join(f"{d}={c}" for d, c in zip(e["dims"], e["config"]))
                print(f"    {dims_str:45s} HR={e['hit_rate']:.1%} n={e['count']:,} p={e['p_value']:.4f}")

        # Results table
        print(f"\n  {'Approach':<15} {'Signals':>8} {'Hit%':>7} {'Long%':>7} {'Short%':>7} {'HiC%':>6} {'HiC_n':>6} {'NoSig':>7} {'Edge':>7}")
        print(f"  {'─' * 80}")
        for key in ["A", "B", "C", "D"]:
            r = result["results"][key]
            edge = r.hr - result["base_rate"] if r.total else 0
            print(f"  {key:<15} {r.total:>8,} {r.hr:>6.1%} {r.long_hr:>6.1%} {r.short_hr:>6.1%} "
                  f"{r.hiconf_hr:>5.1%} {r.hiconf_n:>6,} {r.no_signal:>7,} {edge:>+6.1%}")

        # Stat sig
        for key in ["A", "B", "C"]:
            r = result["results"][key]
            if r.total >= 10:
                try:
                    p = stats.binomtest(r.correct, r.total, result["base_rate"]).pvalue
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    print(f"    {key} p-value: {p:.4f} {sig}")
                except Exception:
                    pass

        # Overfitting check
        for key in ["A", "B"]:
            is_r = result["is_results"][key]
            oos_r = result["results"][key]
            if is_r.total > 0 and oos_r.total > 0:
                deg = is_r.hr - oos_r.hr
                status = "OVERFIT" if deg > 0.05 else "OK"
                print(f"    {key} IS={is_r.hr:.1%} OOS={oos_r.hr:.1%} deg={deg:+.1%} {status}")

        print()

    # ── SUMMARY TABLE ──
    if all_results:
        print()
        print("=" * 90)
        print("SUMMARY — All Targets")
        print("=" * 90)
        print(f"{'Target':>8} {'Base':>6} | {'A_HR':>6} {'A_n':>6} {'A_edge':>7} | {'B_HR':>6} {'B_n':>6} {'B_edge':>7} | {'C_HR':>6} {'C_n':>6} {'C_edge':>7} | {'D_HR':>6}")
        print("─" * 90)
        for res in all_results:
            r = res["results"]
            t = res["target"]
            br = res["base_rate"]
            print(
                f"${t:>6} {br:>5.1%} | "
                f"{r['A'].hr:>5.1%} {r['A'].total:>5,} {r['A'].hr - br:>+6.1%} | "
                f"{r['B'].hr:>5.1%} {r['B'].total:>5,} {r['B'].hr - br:>+6.1%} | "
                f"{r['C'].hr:>5.1%} {r['C'].total:>5,} {r['C'].hr - br:>+6.1%} | "
                f"{r['D'].hr:>5.1%}"
            )
        print("─" * 90)
        print("Edge = Hit Rate - Base Rate (positive = prediction better than always guessing majority)")
        print()

        # Find best
        best_edge = -1
        best_key = ""
        best_target = 0
        for res in all_results:
            for key in ["A", "B", "C"]:
                r = res["results"][key]
                if r.total >= 20:
                    edge = abs(r.hr - res["base_rate"])
                    if edge > best_edge:
                        best_edge = edge
                        best_key = key
                        best_target = res["target"]

        if best_edge > 0:
            print(f"Best edge: Approach {best_key} at ${best_target} target — {best_edge:+.1%} over base rate")


if __name__ == "__main__":
    main()
