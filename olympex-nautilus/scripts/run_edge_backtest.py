#!/usr/bin/env python3
"""
Edge Discovery Backtest — A/B/C/D Test
=======================================
Tìm điểm mà giá sẽ tăng $3 hoặc giảm $3 (ai đến trước),
dùng 4 approach khác nhau, so sánh trên CÙNG dữ liệu.

Approach A: Data-Driven (scan dimensions → tìm vùng có edge)
Approach B: Theory-Driven (regime + transition logic)
Approach C: Combined (B lọc + A tinh chỉnh)
Approach D: Baseline Random (50/50 → benchmark)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 1: MEASUREMENT ENGINE (33-dim state vector)
# ============================================================

SCALES = [5, 10, 20, 50, 100]


def realized_vol(returns: np.ndarray, scale: int) -> float:
    if len(returns) < scale:
        return np.nan
    return np.std(returns[-scale:])


def autocorrelation(returns: np.ndarray, scale: int, lag: int = 1) -> float:
    if len(returns) < scale + lag:
        return np.nan
    r = returns[-scale:]
    if np.std(r) < 1e-12:
        return 0.0
    return np.corrcoef(r[lag:], r[:-lag])[0, 1]


def skewness(returns: np.ndarray, scale: int) -> float:
    if len(returns) < scale:
        return np.nan
    r = returns[-scale:]
    s = np.std(r)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((r - np.mean(r)) / s) ** 3))


def kurtosis(returns: np.ndarray, scale: int) -> float:
    if len(returns) < scale:
        return np.nan
    r = returns[-scale:]
    s = np.std(r)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((r - np.mean(r)) / s) ** 4))


def volume_anomaly(volumes: np.ndarray, scale: int) -> float:
    if len(volumes) < scale:
        return np.nan
    window = volumes[-scale:]
    med = np.median(window)
    if med < 1e-12:
        return 1.0
    return float(volumes[-1] / med)


def pv_correlation(returns: np.ndarray, volumes: np.ndarray, scale: int) -> float:
    if len(returns) < scale or len(volumes) < scale:
        return np.nan
    r = np.abs(returns[-scale:])
    v = volumes[-scale:]
    if np.std(r) < 1e-12 or np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(r, v)[0, 1])


METRIC_FUNCS = {
    "vol": lambda r, v, s: realized_vol(r, s),
    "autocorr": lambda r, v, s: autocorrelation(r, s),
    "skew": lambda r, v, s: skewness(r, s),
    "kurt": lambda r, v, s: kurtosis(r, s),
    "vol_anom": lambda r, v, s: volume_anomaly(v, s),
    "pv_corr": lambda r, v, s: pv_correlation(r, v, s),
}


def compute_base_metrics(returns: np.ndarray, volumes: np.ndarray) -> dict:
    """Compute 6 base metrics, multi-scale consensus."""
    result = {}
    for name, func in METRIC_FUNCS.items():
        values = []
        for s in SCALES:
            val = func(returns, volumes, s)
            if not np.isnan(val):
                values.append(val)
        if values:
            result[name] = float(np.mean(values))
        else:
            result[name] = 0.0
    return result


def compute_state_vector(metric_history: list[dict]) -> dict | None:
    """
    Compute full 33-dim state vector from metric history.
    Returns None if not enough history.
    """
    if len(metric_history) < 3:
        return None

    current = metric_history[-1]
    prev = metric_history[-2]
    prev2 = metric_history[-3]

    state = {}
    dim_names = list(METRIC_FUNCS.keys())

    # Tầng 0: Position (6 dims)
    for name in dim_names:
        state[name] = current[name]

    # Tầng 1: Velocity (6 dims)
    for name in dim_names:
        state[f"d_{name}"] = current[name] - prev[name]

    # Tầng 2: Acceleration (6 dims)
    for name in dim_names:
        v_now = current[name] - prev[name]
        v_prev = prev[name] - prev2[name]
        state[f"dd_{name}"] = v_now - v_prev

    # Tầng 3: Cross-correlation of velocities (15 dims)
    # Use rolling window of velocity history
    if len(metric_history) >= 12:
        vel_history = []
        for i in range(max(0, len(metric_history) - 20), len(metric_history) - 1):
            vel = {}
            for name in dim_names:
                vel[name] = metric_history[i + 1][name] - metric_history[i][name]
            vel_history.append(vel)

        if len(vel_history) >= 5:
            for i, n1 in enumerate(dim_names):
                for n2 in dim_names[i + 1:]:
                    v1 = np.array([v[n1] for v in vel_history])
                    v2 = np.array([v[n2] for v in vel_history])
                    if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
                        state[f"xc_{n1}_{n2}"] = 0.0
                    else:
                        state[f"xc_{n1}_{n2}"] = float(np.corrcoef(v1, v2)[0, 1])
        else:
            for i, n1 in enumerate(dim_names):
                for n2 in dim_names[i + 1:]:
                    state[f"xc_{n1}_{n2}"] = 0.0
    else:
        for i, n1 in enumerate(dim_names):
            for n2 in dim_names[i + 1:]:
                state[f"xc_{n1}_{n2}"] = 0.0

    return state


# ============================================================
# SECTION 2: REGIME MEMBERSHIP (continuous, sum=1)
# ============================================================

def compute_regime_membership(state: dict) -> dict:
    """Compute 4-regime membership (QUIET, MOMENTUM, REVERT, CHAOS)."""
    v = min(max(state.get("vol", 0), 0), 1) if state.get("vol", 0) <= 1 else min(state.get("vol", 0) / 0.01, 1)
    a = np.clip(state.get("autocorr", 0), -1, 1)
    k = min(max((state.get("kurt", 3) - 2) / 6, 0), 1)  # normalize: 3=normal→0.17, >8→1
    va = min(max((state.get("vol_anom", 1) - 0.5) / 3, 0), 1)

    # Raw scores
    quiet = max(0, (1 - v) * (1 - abs(a)) * (1 - k) * (1 - va))
    momentum = max(0, (1 - abs(v - 0.5)) * max(a, 0) * (1 - k))
    revert = max(0, (1 - abs(v - 0.5)) * max(-a, 0) * (1 - k))
    chaos = max(0, v * k * va)

    total = quiet + momentum + revert + chaos
    if total < 1e-12:
        return {"QUIET": 0.25, "MOMENTUM": 0.25, "REVERT": 0.25, "CHAOS": 0.25}

    return {
        "QUIET": quiet / total,
        "MOMENTUM": momentum / total,
        "REVERT": revert / total,
        "CHAOS": chaos / total,
    }


# ============================================================
# SECTION 3: LABELING — $3 UP or $3 DOWN (who gets there first?)
# ============================================================

TARGET_MOVE = 3.0  # dollars
MAX_LOOKAHEAD = 120  # bars (2 hours max)


def label_future_moves(closes: np.ndarray) -> np.ndarray:
    """
    For each bar, determine: does price hit +$3 or -$3 first?
    Returns: array of +1 (up first), -1 (down first), 0 (neither within window)
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        base_price = closes[i]
        target_up = base_price + TARGET_MOVE
        target_down = base_price - TARGET_MOVE

        end = min(i + MAX_LOOKAHEAD + 1, n)
        future = closes[i + 1: end]

        hit_up = -1
        hit_down = -1

        for j, price in enumerate(future):
            if hit_up < 0 and price >= target_up:
                hit_up = j
            if hit_down < 0 and price <= target_down:
                hit_down = j
            if hit_up >= 0 and hit_down >= 0:
                break

        if hit_up >= 0 and hit_down >= 0:
            labels[i] = 1 if hit_up < hit_down else -1
        elif hit_up >= 0:
            labels[i] = 1
        elif hit_down >= 0:
            labels[i] = -1
        else:
            labels[i] = 0  # neither reached

    return labels


# ============================================================
# SECTION 4: APPROACH A — Data-Driven Edge Scanner
# ============================================================

def approach_a_scan(states: list[dict], labels: np.ndarray,
                    train_mask: np.ndarray) -> dict:
    """
    Scan all 33 dimensions, find which ones predict direction.
    Returns: edge conditions (dimension, threshold, direction).
    """
    valid = (labels != 0) & train_mask
    if valid.sum() < 50:
        return {"edges": [], "dims_ranked": []}

    dim_names = list(states[0].keys()) if states else []
    dim_scores = []

    for dim in dim_names:
        vals = np.array([s.get(dim, 0) for s in states])
        dim_vals = vals[valid]
        dim_labels = labels[valid]

        if len(dim_vals) < 50:
            continue

        # Quintile analysis
        try:
            quintiles = np.percentile(dim_vals, [20, 40, 60, 80])
            bins = np.digitize(dim_vals, quintiles)
        except Exception:
            continue

        hit_rates = []
        counts = []
        for b in range(5):
            mask = bins == b
            if mask.sum() < 10:
                hit_rates.append(0.5)
                counts.append(0)
                continue
            hr = (dim_labels[mask] == 1).mean()
            hit_rates.append(hr)
            counts.append(mask.sum())

        # Monotonicity
        if len(hit_rates) >= 4 and min(counts) >= 5:
            rho, _ = stats.spearmanr(range(len(hit_rates)), hit_rates)
            spread = max(hit_rates) - min(hit_rates)
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
    top_dims = [d["dim"] for d in dim_scores[:6]]

    # Find combo edges using top dims
    edges = []
    for combo_size in [1, 2]:
        for combo in combinations(top_dims[:5], combo_size):
            vals_list = [np.array([s.get(d, 0) for s in states]) for d in combo]
            medians = [np.median(v[valid]) for v in vals_list]

            # Try all HIGH/LOW combos
            for bits in range(2 ** combo_size):
                mask = valid.copy()
                config = []
                for ci in range(combo_size):
                    if (bits >> ci) & 1:
                        mask &= (vals_list[ci] > medians[ci])
                        config.append("HIGH")
                    else:
                        mask &= (vals_list[ci] <= medians[ci])
                        config.append("LOW")

                count = mask.sum()
                if count < 25:
                    continue

                hr = (labels[mask] == 1).mean()
                if abs(hr - 0.5) < 0.03:
                    continue

                # Binomial test
                k = int((labels[mask] == 1).sum())
                p_val = stats.binom_test(k, count, 0.5) if hasattr(stats, 'binom_test') else stats.binomtest(k, count, 0.5).pvalue

                edges.append({
                    "dims": list(combo),
                    "config": config,
                    "medians": [float(m) for m in medians],
                    "hit_rate": float(hr),
                    "count": int(count),
                    "p_value": float(p_val),
                    "direction": 1 if hr > 0.5 else -1,
                    "edge_strength": abs(hr - 0.5),
                })

    edges.sort(key=lambda x: x["edge_strength"], reverse=True)
    return {"edges": edges[:20], "dims_ranked": dim_scores[:10]}


def approach_a_predict(state: dict, edges: list[dict]) -> tuple[int, float]:
    """
    Given current state and discovered edges, predict direction.
    Returns (direction, confidence). direction=0 means no signal.
    """
    votes_up = 0.0
    votes_down = 0.0

    for edge in edges:
        match = True
        for i, dim in enumerate(edge["dims"]):
            val = state.get(dim, 0)
            if edge["config"][i] == "HIGH" and val <= edge["medians"][i]:
                match = False
                break
            if edge["config"][i] == "LOW" and val > edge["medians"][i]:
                match = False
                break

        if match:
            weight = edge["edge_strength"]
            if edge["direction"] == 1:
                votes_up += weight
            else:
                votes_down += weight

    if votes_up + votes_down < 0.01:
        return 0, 0.0

    if votes_up > votes_down:
        conf = votes_up / (votes_up + votes_down)
        return 1, conf
    else:
        conf = votes_down / (votes_up + votes_down)
        return -1, conf


# ============================================================
# SECTION 5: APPROACH B — Theory-Driven Signals
# ============================================================

def approach_b_predict(state: dict, membership: dict) -> tuple[int, float]:
    """
    Theory-based prediction using regime + dynamics.
    """
    signals = []

    # B1: Momentum continuation
    if membership["MOMENTUM"] > 0.4 and state.get("autocorr", 0) > 0.05:
        sk = state.get("skew", 0)
        strength = membership["MOMENTUM"] * abs(state.get("autocorr", 0))
        if sk > 0:
            signals.append((1, strength, "momentum_long"))
        elif sk < 0:
            signals.append((-1, strength, "momentum_short"))

    # B2: Mean reversion at extremes
    if membership["REVERT"] > 0.4 and state.get("autocorr", 0) < -0.05:
        sk = state.get("skew", 0)
        strength = membership["REVERT"] * abs(state.get("autocorr", 0))
        if sk > 0.5:
            signals.append((-1, strength * 0.8, "mr_fade_up"))
        elif sk < -0.5:
            signals.append((1, strength * 0.8, "mr_fade_down"))

    # B3: Regime transition QUIET→MOMENTUM
    d_autocorr = state.get("d_autocorr", 0)
    d_vol = state.get("d_vol", 0)
    if membership["QUIET"] > 0.3 and d_autocorr > 0.02 and d_vol > 0:
        sk = state.get("skew", 0)
        strength = d_autocorr * 2
        if sk > 0:
            signals.append((1, strength, "breakout_long"))
        elif sk < 0:
            signals.append((-1, strength, "breakout_short"))

    # B4: CHAOS filter — override everything
    if membership["CHAOS"] > 0.5:
        return 0, 0.0  # No trade in chaos

    # B5: Divergence — volume up but momentum dying
    d_vol_anom = state.get("d_vol_anom", 0)
    if d_vol_anom > 0.1 and d_autocorr < -0.02:
        sk = state.get("skew", 0)
        if sk > 0:
            signals.append((-1, 0.3, "divergence_short"))
        elif sk < 0:
            signals.append((1, 0.3, "divergence_long"))

    if not signals:
        return 0, 0.0

    # Aggregate signals
    total_up = sum(s[1] for s in signals if s[0] == 1)
    total_down = sum(s[1] for s in signals if s[0] == -1)

    if total_up > total_down and total_up > 0.05:
        return 1, min(total_up, 1.0)
    elif total_down > total_up and total_down > 0.05:
        return -1, min(total_down, 1.0)
    return 0, 0.0


# ============================================================
# SECTION 6: APPROACH C — Combined (B filter + A edge)
# ============================================================

def approach_c_predict(state: dict, membership: dict,
                       edges: list[dict]) -> tuple[int, float]:
    """B filters, A refines. Only trade when both agree."""
    b_dir, b_conf = approach_b_predict(state, membership)

    # CHAOS filter from B always respected
    if membership.get("CHAOS", 0) > 0.5:
        return 0, 0.0

    a_dir, a_conf = approach_a_predict(state, edges)

    if a_dir == 0 and b_dir == 0:
        return 0, 0.0

    # Both have signal AND agree?
    if a_dir != 0 and b_dir != 0 and a_dir == b_dir:
        return a_dir, (a_conf + b_conf) / 2

    # Only one has signal — trade with reduced confidence
    if a_dir != 0 and b_dir == 0:
        return a_dir, a_conf * 0.5
    if b_dir != 0 and a_dir == 0:
        return b_dir, b_conf * 0.5

    # Disagree — no trade
    return 0, 0.0


# ============================================================
# SECTION 7: APPROACH D — Baseline (Random)
# ============================================================

def approach_d_predict(rng: np.random.Generator) -> tuple[int, float]:
    """Random prediction — benchmark."""
    r = rng.random()
    if r < 0.33:
        return 1, 0.5
    elif r < 0.66:
        return -1, 0.5
    else:
        return 0, 0.0  # no signal


# ============================================================
# SECTION 8: BACKTEST ENGINE
# ============================================================

@dataclass
class BacktestResult:
    name: str
    total_signals: int = 0
    correct: int = 0
    wrong: int = 0
    no_signal: int = 0
    skipped_no_label: int = 0

    # By direction
    long_signals: int = 0
    long_correct: int = 0
    short_signals: int = 0
    short_correct: int = 0

    # By confidence bucket
    high_conf_signals: int = 0
    high_conf_correct: int = 0
    low_conf_signals: int = 0
    low_conf_correct: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.correct / self.total_signals

    @property
    def long_hit_rate(self) -> float:
        if self.long_signals == 0:
            return 0.0
        return self.long_correct / self.long_signals

    @property
    def short_hit_rate(self) -> float:
        if self.short_signals == 0:
            return 0.0
        return self.short_correct / self.short_signals

    @property
    def high_conf_hit_rate(self) -> float:
        if self.high_conf_signals == 0:
            return 0.0
        return self.high_conf_correct / self.high_conf_signals

    @property
    def low_conf_hit_rate(self) -> float:
        if self.low_conf_signals == 0:
            return 0.0
        return self.low_conf_correct / self.low_conf_signals


def run_backtest():
    """Main backtest runner."""
    # ── Load data ──
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)
    print(f"Loaded {n:,} candles")
    print(f"Price range: ${closes.min():,.0f} — ${closes.max():,.0f}")
    print(f"Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print()

    # ── Compute returns ──
    returns = np.diff(closes) / closes[:-1]
    returns = np.concatenate([[0.0], returns])

    # ── Label: +$3 or -$3 first ──
    print(f"Labeling future ${TARGET_MOVE} moves (max {MAX_LOOKAHEAD} bars lookahead)...")
    labels = label_future_moves(closes)
    n_up = (labels == 1).sum()
    n_down = (labels == -1).sum()
    n_neither = (labels == 0).sum()
    print(f"  UP first:   {n_up:,} ({n_up / n * 100:.1f}%)")
    print(f"  DOWN first: {n_down:,} ({n_down / n * 100:.1f}%)")
    print(f"  Neither:    {n_neither:,} ({n_neither / n * 100:.1f}%)")
    print(f"  Base rate (UP among UP+DOWN): {n_up / max(n_up + n_down, 1) * 100:.1f}%")
    print()

    # ── Compute state vectors ──
    print("Computing 33-dim state vectors...")
    WARMUP = 110  # need at least 100 bars + margin
    metric_history = []
    states = [None] * n

    for i in range(WARMUP, n):
        r_window = returns[max(0, i - 200): i + 1]
        v_window = volumes[max(0, i - 200): i + 1]

        metrics = compute_base_metrics(r_window, v_window)
        metric_history.append(metrics)

        state = compute_state_vector(metric_history)
        if state is not None:
            states[i] = state

    valid_states = sum(1 for s in states if s is not None)
    print(f"  Valid state vectors: {valid_states:,}")
    print()

    # ── Train/Test split (67/33) ──
    split_idx = int(n * 0.67)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if states[i] is not None:
            if i < split_idx:
                train_mask[i] = True
            else:
                test_mask[i] = True

    print(f"Train: {train_mask.sum():,} bars | Test: {test_mask.sum():,} bars")
    print(f"Split at index {split_idx:,}")
    print()

    # ── Approach A: Train on in-sample ──
    print("=" * 60)
    print("TRAINING Approach A (Data-Driven)...")
    train_states = [states[i] if states[i] is not None else {} for i in range(n)]
    a_result = approach_a_scan(train_states, labels, train_mask)
    n_edges = len(a_result["edges"])
    print(f"  Found {n_edges} edges")
    if a_result["dims_ranked"]:
        print(f"  Top predictive dimensions:")
        for d in a_result["dims_ranked"][:5]:
            hr_str = " ".join(f"{h:.1%}" for h in d["hit_rates"])
            print(f"    {d['dim']:20s}  mono={d['monotonicity']:+.2f}  spread={d['spread']:.3f}  [{hr_str}]")
    if a_result["edges"]:
        print(f"  Top edges:")
        for e in a_result["edges"][:5]:
            dims_str = " & ".join(f"{d}={c}" for d, c in zip(e["dims"], e["config"]))
            print(f"    {dims_str:40s}  HR={e['hit_rate']:.1%}  n={e['count']:,}  p={e['p_value']:.4f}")
    print()

    # ── Run all 4 approaches on TEST set ──
    print("=" * 60)
    print("BACKTESTING on out-of-sample test set...")
    print("=" * 60)
    print()

    results = {
        "A": BacktestResult("A: Data-Driven"),
        "B": BacktestResult("B: Theory-Driven"),
        "C": BacktestResult("C: Combined"),
        "D": BacktestResult("D: Random Baseline"),
    }

    rng = np.random.default_rng(42)

    test_indices = np.where(test_mask)[0]
    for i in test_indices:
        if labels[i] == 0:
            for r in results.values():
                r.skipped_no_label += 1
            continue

        state = states[i]
        if state is None:
            continue

        actual = labels[i]  # +1 or -1
        membership = compute_regime_membership(state)

        # Approach A
        a_dir, a_conf = approach_a_predict(state, a_result["edges"])
        _record_prediction(results["A"], a_dir, a_conf, actual)

        # Approach B
        b_dir, b_conf = approach_b_predict(state, membership)
        _record_prediction(results["B"], b_dir, b_conf, actual)

        # Approach C
        c_dir, c_conf = approach_c_predict(state, membership, a_result["edges"])
        _record_prediction(results["C"], c_dir, c_conf, actual)

        # Approach D
        d_dir, d_conf = approach_d_predict(rng)
        _record_prediction(results["D"], d_dir, d_conf, actual)

    # ── Print results ──
    print_results(results, n_up, n_down)

    # ── Also run on TRAIN set to show overfitting ──
    print()
    print("=" * 60)
    print("IN-SAMPLE (train set) — for overfitting check:")
    print("=" * 60)
    print()

    train_results = {
        "A": BacktestResult("A: Data-Driven (train)"),
        "B": BacktestResult("B: Theory-Driven (train)"),
    }

    train_indices = np.where(train_mask)[0]
    for i in train_indices:
        if labels[i] == 0:
            continue
        state = states[i]
        if state is None:
            continue
        actual = labels[i]
        membership = compute_regime_membership(state)

        a_dir, a_conf = approach_a_predict(state, a_result["edges"])
        _record_prediction(train_results["A"], a_dir, a_conf, actual)

        b_dir, b_conf = approach_b_predict(state, membership)
        _record_prediction(train_results["B"], b_dir, b_conf, actual)

    for key in ["A", "B"]:
        r = train_results[key]
        oos = results[key]
        print(f"{r.name}")
        print(f"  In-sample HR:  {r.hit_rate:.1%} ({r.total_signals:,} signals)")
        print(f"  Out-of-sample: {oos.hit_rate:.1%} ({oos.total_signals:,} signals)")
        deg = r.hit_rate - oos.hit_rate
        print(f"  Degradation:   {deg:+.1%} {'⚠️  OVERFITTING' if deg > 0.05 else '✓ OK'}")
        print()


def _record_prediction(result: BacktestResult, direction: int, confidence: float, actual: int):
    """Record a single prediction result."""
    if direction == 0:
        result.no_signal += 1
        return

    result.total_signals += 1
    correct = (direction == actual)

    if correct:
        result.correct += 1
    else:
        result.wrong += 1

    if direction == 1:
        result.long_signals += 1
        if correct:
            result.long_correct += 1
    else:
        result.short_signals += 1
        if correct:
            result.short_correct += 1

    if confidence > 0.6:
        result.high_conf_signals += 1
        if correct:
            result.high_conf_correct += 1
    else:
        result.low_conf_signals += 1
        if correct:
            result.low_conf_correct += 1


def print_results(results: dict, n_up: int, n_down: int):
    """Print comparison table."""
    print()
    base_rate = n_up / max(n_up + n_down, 1)

    # Header
    print(f"{'Approach':<25} {'Signals':>8} {'Hit%':>7} {'Long%':>7} {'Short%':>7} {'HiConf%':>8} {'NoSig':>7} {'Edge':>7}")
    print("─" * 85)

    for key in ["A", "B", "C", "D"]:
        r = results[key]
        edge = r.hit_rate - base_rate if r.total_signals > 0 else 0
        print(
            f"{r.name:<25} "
            f"{r.total_signals:>8,} "
            f"{r.hit_rate:>6.1%} "
            f"{r.long_hit_rate:>6.1%} "
            f"{r.short_hit_rate:>6.1%} "
            f"{r.high_conf_hit_rate:>7.1%} "
            f"{r.no_signal:>7,} "
            f"{edge:>+6.1%}"
        )

    print("─" * 85)
    print(f"  Base rate (UP): {base_rate:.1%} | If always guess UP → {base_rate:.1%}")
    print(f"  Edge = Hit Rate - Base Rate (positive = better than always guessing majority)")
    print()

    # Statistical significance
    print("Statistical Significance (binomial test vs base rate):")
    for key in ["A", "B", "C"]:
        r = results[key]
        if r.total_signals > 0:
            p = stats.binomtest(r.correct, r.total_signals, base_rate).pvalue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {r.name}: p={p:.4f} {sig}")
    print()

    # Detailed breakdown
    for key in ["A", "B", "C", "D"]:
        r = results[key]
        print(f"── {r.name} ──")
        print(f"  Total signals:    {r.total_signals:,}")
        print(f"  Correct:          {r.correct:,}  ({r.hit_rate:.1%})")
        print(f"  Wrong:            {r.wrong:,}")
        print(f"  No signal:        {r.no_signal:,}")
        print(f"  Skipped (no $3):  {r.skipped_no_label:,}")
        print(f"  LONG:  {r.long_signals:,} signals, {r.long_hit_rate:.1%} hit rate")
        print(f"  SHORT: {r.short_signals:,} signals, {r.short_hit_rate:.1%} hit rate")
        print(f"  High confidence (>60%): {r.high_conf_signals:,} signals, {r.high_conf_hit_rate:.1%} hit rate")
        print(f"  Low confidence (≤60%):  {r.low_conf_signals:,} signals, {r.low_conf_hit_rate:.1%} hit rate")
        print()


if __name__ == "__main__":
    run_backtest()
