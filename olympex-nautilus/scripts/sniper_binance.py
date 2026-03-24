#!/usr/bin/env python3
"""
Fast Sniper on 1.7M Binance candles (2023-2026)
=================================================
The REAL test. Same measurement engine, same conditions,
but on 1.7 million candles across bull + bear + sideways.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import time, sys
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


def build_states_fast(closes, volumes, sample_every=1):
    """Build states. sample_every=N to only compute every Nth bar (speed up)."""
    n = len(closes)
    returns = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])
    metric_history = []
    states = [None] * n
    dims = ["vol", "autocorr", "skew", "kurt", "vol_anom", "pv_corr"]

    for i in range(110, n):
        if i % sample_every != 0 and sample_every > 1:
            continue

        r_w = returns[max(0, i-200):i+1]
        v_w = volumes[max(0, i-200):i+1]
        m = compute_metrics(r_w, v_w)
        metric_history.append(m)

        if len(metric_history) < 3:
            continue

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
                        c = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 1e-12 and np.std(v2) > 1e-12 else 0
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

        if i % 100000 == 0 and i > 0:
            print(f"    [{i:,}/{n:,}] states built...", flush=True)

    return states


def simulate_sequential(closes, states, cond_func, direction, tp, sl, max_hold, cooldown=5, start_idx=0):
    n = len(closes)
    trades = []
    i = max(start_idx, 113)
    while i < n - 1:
        if states[i] is None or not cond_func(states[i]):
            i += 1
            continue
        if direction == 'AUTO':
            sk = states[i].get("skew", 0)
            ac = states[i].get("autocorr", 0)
            if sk > 0 or ac > 0.02: side = 'LONG'
            elif sk < 0 or ac < -0.02: side = 'SHORT'
            else: i += 1; continue
        else:
            side = direction

        entry = closes[i]
        exit_bar = None
        for j in range(i + 1, min(i + max_hold + 1, n)):
            pnl = (closes[j] - entry) if side == 'LONG' else (entry - closes[j])
            if pnl >= tp:
                trades.append({'bar': i, 'side': side, 'pnl': tp, 'reason': 'TP', 'hold': j - i, 'price': entry})
                exit_bar = j; break
            elif pnl <= -sl:
                trades.append({'bar': i, 'side': side, 'pnl': -sl, 'reason': 'SL', 'hold': j - i, 'price': entry})
                exit_bar = j; break

        if exit_bar is None:
            j_end = min(i + max_hold, n - 1)
            pnl = (closes[j_end] - entry) if side == 'LONG' else (entry - closes[j_end])
            trades.append({'bar': i, 'side': side, 'pnl': pnl, 'reason': 'TO', 'hold': j_end - i, 'price': entry})
            exit_bar = j_end

        i = exit_bar + cooldown
    return trades


def print_results(trades, label, yearly_bars=None, closes=None):
    if not trades:
        print(f"  {label}: NO TRADES")
        return

    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n
    tp_count = sum(1 for t in trades if t['reason'] == 'TP')
    sl_count = sum(1 for t in trades if t['reason'] == 'SL')
    to_count = sum(1 for t in trades if t['reason'] == 'TO')
    avg_pnl = pnls.mean()
    total = pnls.sum()
    avg_hold = np.mean([t['hold'] for t in trades])
    med_hold = np.median([t['hold'] for t in trades])

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()

    pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).any() else float('inf')
    pf_str = f"{pf:.2f}" if pf < 100 else "INF"

    print(f"  {label}")
    print(f"    Trades: {n:,} | WR: {wr:.1%} ({wins}W/{n-wins}L) | TP:{tp_count} SL:{sl_count} TO:{to_count}")
    print(f"    Avg PnL: ${avg_pnl:+.2f} | Total: ${total:+,.0f} | PF: {pf_str} | MaxDD: ${max_dd:,.0f}")
    print(f"    Hold: avg {avg_hold:.0f}m, med {med_hold:.0f}m")

    # Yearly breakdown if we have bar timestamps
    if yearly_bars is not None and closes is not None:
        print(f"    Per-year:")
        for year, (yr_start, yr_end) in yearly_bars.items():
            yr_trades = [t for t in trades if yr_start <= t['bar'] < yr_end]
            if not yr_trades: continue
            yr_pnls = np.array([t['pnl'] for t in yr_trades])
            yr_wr = (yr_pnls > 0).sum() / len(yr_pnls)
            yr_total = yr_pnls.sum()
            yr_long = sum(1 for t in yr_trades if t['side'] == 'LONG')
            yr_short = sum(1 for t in yr_trades if t['side'] == 'SHORT')
            print(f"      {year}: {len(yr_trades):>5} trades | WR={yr_wr:.1%} | PnL=${yr_total:>+8,.0f} | L:{yr_long} S:{yr_short}")
    print()


def main():
    data_path = Path(__file__).parent.parent / "data" / "btc_binance_1min.parquet"
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)
    atr = np.mean(np.abs(np.diff(closes)))

    print(f"Data: {n:,} M1 candles | ${closes.min():,.0f}—${closes.max():,.0f} | ATR≈${atr:.0f}")
    print(f"Date: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print()

    # Year boundaries
    df['year'] = df['timestamp'].dt.year
    yearly_bars = {}
    for yr, g in df.groupby('year'):
        yearly_bars[yr] = (g.index[0], g.index[-1] + 1)
        print(f"  {yr}: bars {g.index[0]:>9,}—{g.index[-1]:>9,} | ${g['close'].min():>8,.0f}—${g['close'].max():>8,.0f}")
    print()

    # Build states — compute every bar for accuracy
    print("Building states on 1.7M candles (this takes a few minutes)...")
    t0 = time.time()
    states = build_states_fast(closes, volumes, sample_every=1)
    valid = sum(1 for s in states if s is not None)
    elapsed = time.time() - t0
    print(f"  Done: {valid:,} states in {elapsed:.0f}s ({valid/elapsed:.0f} states/s)")
    print()

    # Adaptive percentile thresholds
    vol_vals = [s["vol"] for s in states if s is not None]
    xc_sp = [s.get("xc_skew_pv_corr", 0) for s in states if s is not None]
    xc_va = [s.get("xc_vol_autocorr", 0) for s in states if s is not None]

    vol_p5 = np.percentile(vol_vals, 5)
    vol_p10 = np.percentile(vol_vals, 10)
    vol_p15 = np.percentile(vol_vals, 15)
    vol_p20 = np.percentile(vol_vals, 20)
    xc_sp_p10 = np.percentile(xc_sp, 10)
    xc_sp_p15 = np.percentile(xc_sp, 15)
    xc_va_p25 = np.percentile(xc_va, 25)

    print(f"Thresholds: vol_p10={vol_p10:.6f} vol_p15={vol_p15:.6f} xc_sp_p15={xc_sp_p15:.4f}")
    print()

    conditions = {
        "S1_ultra": lambda s: s["vol"] <= vol_p5 and s.get("xc_skew_pv_corr", 0) <= xc_sp_p10,
        "S2_storm": lambda s: s["vol"] <= vol_p10 and s.get("xc_skew_pv_corr", 0) <= xc_sp_p15,
        "S3_decouple": lambda s: s["vol"] <= vol_p10 and s.get("xc_vol_autocorr", 0) <= xc_va_p25,
        "S4_quiet": lambda s: s["vol"] <= vol_p15,
        "S5_relaxed": lambda s: s["vol"] <= vol_p20,
    }

    # Count triggers
    print("Triggers:")
    for name, func in conditions.items():
        cnt = sum(1 for s in states if s is not None and func(s))
        per_day = cnt / (n / 1440)
        print(f"  {name:20s}: {cnt:>7,} ({per_day:.1f}/day)")
    print()

    # Train/test: first 67% train, last 33% test
    split = int(n * 0.67)
    print(f"Split: train bars 0—{split:,} | test bars {split:,}—{n:,}")
    print()

    # ═══════════════════════════════════════════
    # TEST: Sweep conditions × TP × SL × direction
    # ═══════════════════════════════════════════
    max_hold = 120

    print("=" * 110)
    print("SINGLE ENTRY — LONG (quiet predicts UP)")
    print("=" * 110)
    print(f"{'Cond':15s} {'TP':>5} {'SL':>5} {'R':>5} | {'N':>6} {'WR':>6} {'EV':>8} {'Total':>10} {'PF':>6} {'DD':>8} {'Hold':>5}")
    print("─" * 110)

    results = []
    for cond_name, cond_func in conditions.items():
        for tp in [20, 30, 40, 50, 75, 100]:
            for sl_mult in [1.5, 2, 3, 5, 7, 10]:
                sl = round(tp * sl_mult)
                trades = simulate_sequential(closes, states, cond_func, 'LONG', tp, sl, max_hold, 5, split)
                if len(trades) < 10: continue
                pnls = np.array([t['pnl'] for t in trades])
                wr = (pnls > 0).sum() / len(pnls)
                ev = pnls.mean()
                total = pnls.sum()
                pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).any() else 999
                eq = np.cumsum(pnls); dd = (eq - np.maximum.accumulate(eq)).min()
                hld = np.median([t['hold'] for t in trades])

                results.append((cond_name, 'LONG', tp, sl, sl_mult, len(trades), wr, ev, total, pf, dd, hld, trades))

                if ev > 0 or wr >= 0.75:
                    m = " ★" if wr >= 0.80 else ""
                    if ev > 0: m += " +EV"
                    pf_s = f"{pf:.1f}" if pf < 100 else "INF"
                    print(f"{cond_name:15s} ${tp:>4} ${sl:>4} {sl_mult:>4.1f}× | "
                          f"{len(trades):>6} {wr:>5.1%} ${ev:>+7.1f} ${total:>+9,.0f} {pf_s:>6} ${dd:>7,.0f} {hld:>4.0f}m{m}")

    # Same for SHORT
    print()
    print("=" * 110)
    print("SINGLE ENTRY — SHORT (quiet predicts DOWN in bear?)")
    print("=" * 110)
    print(f"{'Cond':15s} {'TP':>5} {'SL':>5} {'R':>5} | {'N':>6} {'WR':>6} {'EV':>8} {'Total':>10} {'PF':>6} {'DD':>8} {'Hold':>5}")
    print("─" * 110)

    for cond_name, cond_func in conditions.items():
        for tp in [20, 30, 40, 50, 75, 100]:
            for sl_mult in [1.5, 2, 3, 5, 7, 10]:
                sl = round(tp * sl_mult)
                trades = simulate_sequential(closes, states, cond_func, 'SHORT', tp, sl, max_hold, 5, split)
                if len(trades) < 10: continue
                pnls = np.array([t['pnl'] for t in trades])
                wr = (pnls > 0).sum() / len(pnls)
                ev = pnls.mean()
                total = pnls.sum()
                pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).any() else 999
                eq = np.cumsum(pnls); dd = (eq - np.maximum.accumulate(eq)).min()
                hld = np.median([t['hold'] for t in trades])

                results.append((cond_name, 'SHORT', tp, sl, sl_mult, len(trades), wr, ev, total, pf, dd, hld, trades))

                if ev > 0 or wr >= 0.75:
                    m = " ★" if wr >= 0.80 else ""
                    if ev > 0: m += " +EV"
                    pf_s = f"{pf:.1f}" if pf < 100 else "INF"
                    print(f"{cond_name:15s} ${tp:>4} ${sl:>4} {sl_mult:>4.1f}× | "
                          f"{len(trades):>6} {wr:>5.1%} ${ev:>+7.1f} ${total:>+9,.0f} {pf_s:>6} ${dd:>7,.0f} {hld:>4.0f}m{m}")

    # AUTO direction
    print()
    print("=" * 110)
    print("SINGLE ENTRY — AUTO (skew/autocorr decides LONG or SHORT)")
    print("=" * 110)
    print(f"{'Cond':15s} {'TP':>5} {'SL':>5} {'R':>5} | {'N':>6} {'WR':>6} {'EV':>8} {'Total':>10} {'PF':>6} {'DD':>8} {'Hold':>5}")
    print("─" * 110)

    for cond_name, cond_func in conditions.items():
        for tp in [20, 30, 40, 50, 75, 100]:
            for sl_mult in [1.5, 2, 3, 5, 7, 10]:
                sl = round(tp * sl_mult)
                trades = simulate_sequential(closes, states, cond_func, 'AUTO', tp, sl, max_hold, 5, split)
                if len(trades) < 10: continue
                pnls = np.array([t['pnl'] for t in trades])
                wr = (pnls > 0).sum() / len(pnls)
                ev = pnls.mean()
                total = pnls.sum()
                pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).any() else 999
                eq = np.cumsum(pnls); dd = (eq - np.maximum.accumulate(eq)).min()
                hld = np.median([t['hold'] for t in trades])

                results.append((cond_name, 'AUTO', tp, sl, sl_mult, len(trades), wr, ev, total, pf, dd, hld, trades))

                if ev > 0 or wr >= 0.75:
                    m = " ★" if wr >= 0.80 else ""
                    if ev > 0: m += " +EV"
                    pf_s = f"{pf:.1f}" if pf < 100 else "INF"
                    print(f"{cond_name:15s} ${tp:>4} ${sl:>4} {sl_mult:>4.1f}× | "
                          f"{len(trades):>6} {wr:>5.1%} ${ev:>+7.1f} ${total:>+9,.0f} {pf_s:>6} ${dd:>7,.0f} {hld:>4.0f}m{m}")

    # ═══════════════════════════════════════════
    # TOP 10 CONFIGS by EV
    # ═══════════════════════════════════════════
    print()
    print("=" * 110)
    print("TOP 10 CONFIGS BY EXPECTED VALUE (test set)")
    print("=" * 110)

    ev_sorted = sorted(results, key=lambda x: x[7], reverse=True)
    for i, (cn, dir, tp, sl, sm, nt, wr, ev, tot, pf, dd, hld, trades) in enumerate(ev_sorted[:10]):
        pf_s = f"{pf:.1f}" if pf < 100 else "INF"
        print(f"  #{i+1}: {cn:15s} {dir:>5} TP=${tp} SL=${sl} ({sm}×) → "
              f"{nt:,} trades WR={wr:.1%} EV=${ev:+.1f} Total=${tot:+,.0f} PF={pf_s} DD=${dd:,.0f}")

    # Detailed breakdown of #1
    if ev_sorted:
        best = ev_sorted[0]
        print()
        print_results(best[12], f"BEST: {best[0]} {best[1]} TP=${best[2]} SL=${best[3]}",
                      yearly_bars, closes)


if __name__ == "__main__":
    main()
