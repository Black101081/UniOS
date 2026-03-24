#!/usr/bin/env python3
"""
Fast Sniper — Vào nhanh, ra nhanh, WR cực cao
================================================

Bài toán:
  - Dùng 33-dim state vector tìm điểm entry WR cao nhất
  - 1 lệnh tại 1 thời điểm (sequential, không overlap)
  - TP = $X, SL = $Y (sweep cả 2)
  - Đo: WR, EV, avg hold time, max DD, profit factor
  - So sánh Single (1 hướng) vs Dual (cả 2 hướng)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations
import time
import warnings
warnings.filterwarnings("ignore")

SCALES = [5, 10, 20, 50, 100]

# ── Measurement Engine (compact) ──
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


# ── Simulate sequential trades ──

def simulate_sequential(closes, states, condition_func, direction, tp, sl, max_hold,
                        cooldown=5, start_idx=0):
    """
    Simulate sequential trades (1 at a time).
    direction: 'LONG', 'SHORT', or 'AUTO' (use skew to decide)
    cooldown: minimum bars between trades
    """
    n = len(closes)
    trades = []
    i = max(start_idx, 113)  # after warmup

    while i < n - 1:
        # Check condition
        if states[i] is None or not condition_func(states[i]):
            i += 1
            continue

        # Determine direction
        if direction == 'AUTO':
            sk = states[i].get("skew", 0)
            ac = states[i].get("autocorr", 0)
            if sk > 0 or ac > 0.02:
                side = 'LONG'
            elif sk < 0 or ac < -0.02:
                side = 'SHORT'
            else:
                i += 1
                continue
        else:
            side = direction

        entry_price = closes[i]
        entry_bar = i

        # Walk forward to find exit
        exit_price = None
        exit_reason = None
        exit_bar = None

        for j in range(i + 1, min(i + max_hold + 1, n)):
            price = closes[j]

            if side == 'LONG':
                pnl = price - entry_price
                if pnl >= tp:
                    exit_price = entry_price + tp
                    exit_reason = 'TP'
                    exit_bar = j
                    break
                elif pnl <= -sl:
                    exit_price = entry_price - sl
                    exit_reason = 'SL'
                    exit_bar = j
                    break
            else:  # SHORT
                pnl = entry_price - price
                if pnl >= tp:
                    exit_price = entry_price - tp
                    exit_reason = 'TP'
                    exit_bar = j
                    break
                elif pnl <= -sl:
                    exit_price = entry_price + sl
                    exit_reason = 'SL'
                    exit_bar = j
                    break

        if exit_bar is None:
            # Max hold reached — close at market
            exit_bar = min(i + max_hold, n - 1)
            exit_price = closes[exit_bar]
            if side == 'LONG':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            exit_reason = 'TIMEOUT'
        else:
            pnl = tp if exit_reason == 'TP' else -sl

        hold_time = exit_bar - entry_bar
        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': side,
            'pnl': pnl,
            'reason': exit_reason,
            'hold_time': hold_time,
        })

        # Next entry after cooldown
        i = exit_bar + cooldown

    return trades


def simulate_dual(closes, states, condition_func, tp, sl, max_hold, cooldown=5, start_idx=0):
    """
    Dual entry: LONG + SHORT simultaneously.
    Close individually when each hits TP or SL.
    """
    n = len(closes)
    trades = []
    i = max(start_idx, 113)

    while i < n - 1:
        if states[i] is None or not condition_func(states[i]):
            i += 1
            continue

        entry_price = closes[i]
        entry_bar = i

        long_pnl = None
        long_exit = None
        long_reason = None
        short_pnl = None
        short_exit = None
        short_reason = None

        for j in range(i + 1, min(i + max_hold + 1, n)):
            price = closes[j]

            # LONG side
            if long_pnl is None:
                lpnl = price - entry_price
                if lpnl >= tp:
                    long_pnl = tp
                    long_exit = j
                    long_reason = 'TP'
                elif lpnl <= -sl:
                    long_pnl = -sl
                    long_exit = j
                    long_reason = 'SL'

            # SHORT side
            if short_pnl is None:
                spnl = entry_price - price
                if spnl >= tp:
                    short_pnl = tp
                    short_exit = j
                    short_reason = 'TP'
                elif spnl <= -sl:
                    short_pnl = -sl
                    short_exit = j
                    short_reason = 'SL'

            if long_pnl is not None and short_pnl is not None:
                break

        # Timeout for remaining open
        if long_pnl is None:
            j_end = min(i + max_hold, n - 1)
            long_pnl = closes[j_end] - entry_price
            long_exit = j_end
            long_reason = 'TIMEOUT'
        if short_pnl is None:
            j_end = min(i + max_hold, n - 1)
            short_pnl = entry_price - closes[j_end]
            short_exit = j_end
            short_reason = 'TIMEOUT'

        total_pnl = long_pnl + short_pnl
        last_exit = max(long_exit, short_exit)

        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': last_exit,
            'entry_price': entry_price,
            'long_pnl': long_pnl,
            'long_reason': long_reason,
            'short_pnl': short_pnl,
            'short_reason': short_reason,
            'total_pnl': total_pnl,
            'hold_time': last_exit - entry_bar,
        })

        i = last_exit + cooldown

    return trades


def analyze_trades(trades, label):
    if not trades:
        return None

    pnls = [t.get('pnl', t.get('total_pnl', 0)) for t in trades]
    n = len(pnls)
    pnls = np.array(pnls)

    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    wr = wins / n if n > 0 else 0

    avg_pnl = pnls.mean()
    total_pnl = pnls.sum()
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls < 0].mean() if losses > 0 else 0
    profit_factor = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if losses > 0 and pnls[pnls < 0].sum() != 0 else float('inf')

    # Drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()

    # Hold time
    hold_times = [t['hold_time'] for t in trades]
    avg_hold = np.mean(hold_times)
    med_hold = np.median(hold_times)

    # Win/loss by reason
    if 'reason' in trades[0]:
        tp_count = sum(1 for t in trades if t['reason'] == 'TP')
        sl_count = sum(1 for t in trades if t['reason'] == 'SL')
        to_count = sum(1 for t in trades if t['reason'] == 'TIMEOUT')
    else:
        tp_count = sl_count = to_count = 0

    return {
        'label': label, 'n': n, 'wins': wins, 'losses': losses, 'wr': wr,
        'avg_pnl': avg_pnl, 'total_pnl': total_pnl,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'profit_factor': profit_factor, 'max_dd': max_dd,
        'avg_hold': avg_hold, 'med_hold': med_hold,
        'tp_count': tp_count, 'sl_count': sl_count, 'to_count': to_count,
    }


def main():
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)
    atr = np.mean(np.abs(np.diff(closes)))

    print(f"Data: {n:,} M1 candles | ${closes.min():,.0f}—${closes.max():,.0f} | ATR≈${atr:.0f}")
    print()

    print("Building states...")
    t0 = time.time()
    states = build_states(closes, volumes)
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    # ── Adaptive thresholds ──
    vol_vals = [s["vol"] for s in states if s is not None]
    kurt_vals = [s["kurt"] for s in states if s is not None]
    pv_vals = [s["pv_corr"] for s in states if s is not None]
    xc_sp_vals = [s.get("xc_skew_pv_corr", 0) for s in states if s is not None]
    xc_va_vals = [s.get("xc_vol_autocorr", 0) for s in states if s is not None]

    vol_p5 = np.percentile(vol_vals, 5)
    vol_p10 = np.percentile(vol_vals, 10)
    vol_p15 = np.percentile(vol_vals, 15)
    vol_p80 = np.percentile(vol_vals, 80)
    vol_p90 = np.percentile(vol_vals, 90)
    kurt_p80 = np.percentile(kurt_vals, 80)
    xc_sp_p10 = np.percentile(xc_sp_vals, 10)
    xc_sp_p15 = np.percentile(xc_sp_vals, 15)
    xc_va_p25 = np.percentile(xc_va_vals, 25)
    pv_p25 = np.percentile(pv_vals, 25)

    # ── Conditions ranked by strictness ──
    conditions = {
        # Ultra strict → highest WR, fewer trades
        "S1_ultra_quiet": lambda s: s["vol"] <= vol_p5 and s.get("xc_skew_pv_corr", 0) <= xc_sp_p10,
        # Strict
        "S2_quiet_storm": lambda s: s["vol"] <= vol_p10 and s.get("xc_skew_pv_corr", 0) <= xc_sp_p15,
        # Medium
        "S3_quiet_decouple": lambda s: s["vol"] <= vol_p10 and s.get("xc_vol_autocorr", 0) <= xc_va_p25,
        # Relaxed (more trades)
        "S4_just_quiet": lambda s: s["vol"] <= vol_p15,
        # High vol (for comparison — should perform differently)
        "S5_vol_accel": lambda s: s.get("d_vol", 0) > 0 and s["vol"] >= vol_p80,
        # Chaos (for dual entry)
        "S6_chaos": lambda s: s["vol"] >= vol_p90 and s["kurt"] >= kurt_p80,
    }

    # Count triggers
    print("Condition triggers:")
    for name, func in conditions.items():
        count = sum(1 for s in states if s is not None and func(s))
        per_day = count / (n / 1440)
        print(f"  {name:25s}: {count:>5} ({per_day:.1f}/day)")
    print()

    # ══════════════════════════════════════════════════════
    # PART 1: SINGLE ENTRY — Sweep TP × SL × Condition
    # ══════════════════════════════════════════════════════
    print("=" * 100)
    print("PART 1: SINGLE ENTRY (BUY only — quiet conditions predict UP)")
    print("=" * 100)
    print()

    tp_values = [20, 30, 40, 50, 60, 75, 100]
    sl_multipliers = [1, 1.5, 2, 3, 5, 7, 10]
    max_hold = 120  # 2 hours max

    # Split: train first 67%, test last 33%
    split = int(n * 0.67)

    print(f"{'Condition':25s} {'TP':>5} {'SL':>5} {'SL/TP':>6} | {'Trades':>6} {'WR':>6} {'AvgPnL':>8} {'Total':>9} "
          f"{'PF':>5} {'MaxDD':>8} {'AvgHold':>8} {'TP%':>5} {'SL%':>5} {'TO%':>5}")
    print("─" * 130)

    best_ev = -999
    best_config = None

    for cond_name in ["S1_ultra_quiet", "S2_quiet_storm", "S3_quiet_decouple", "S4_just_quiet"]:
        cond_func = conditions[cond_name]

        for tp in tp_values:
            for sl_mult in sl_multipliers:
                sl = round(tp * sl_mult)

                # Test set only
                trades = simulate_sequential(closes, states, cond_func, 'LONG', tp, sl, max_hold,
                                             cooldown=5, start_idx=split)
                if len(trades) < 5:
                    continue

                r = analyze_trades(trades, f"{cond_name} TP={tp} SL={sl}")
                if r is None:
                    continue

                marker = ""
                if r['wr'] >= 0.90:
                    marker = " ★★★"
                elif r['wr'] >= 0.85:
                    marker = " ★★"
                elif r['wr'] >= 0.80:
                    marker = " ★"

                # Only print interesting ones
                if r['wr'] >= 0.70 or r['total_pnl'] > 0:
                    tp_pct = r['tp_count'] / r['n'] * 100
                    sl_pct = r['sl_count'] / r['n'] * 100
                    to_pct = r['to_count'] / r['n'] * 100
                    pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "INF"

                    print(f"{cond_name:25s} ${tp:>4} ${sl:>4} {sl_mult:>5.1f}× | "
                          f"{r['n']:>6} {r['wr']:>5.1%} ${r['avg_pnl']:>+7.1f} ${r['total_pnl']:>+8.0f} "
                          f"{pf_str:>5} ${r['max_dd']:>7.0f} {r['med_hold']:>7.0f}m "
                          f"{tp_pct:>4.0f}% {sl_pct:>4.0f}% {to_pct:>4.0f}%{marker}")

                    if r['avg_pnl'] > best_ev and r['n'] >= 10:
                        best_ev = r['avg_pnl']
                        best_config = (cond_name, tp, sl, sl_mult, r)

    if best_config:
        name, tp, sl, mult, r = best_config
        print(f"\n  ★ BEST SINGLE: {name} TP=${tp} SL=${sl} ({mult}×)")
        print(f"    {r['n']} trades | WR={r['wr']:.1%} | EV=${r['avg_pnl']:+.1f} | Total=${r['total_pnl']:+,.0f} | MaxDD=${r['max_dd']:,.0f}")

    # ══════════════════════════════════════════════════════
    # PART 2: DUAL ENTRY — High vol conditions
    # ══════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("PART 2: DUAL ENTRY (LONG+SHORT simultaneous — volatility conditions)")
    print("=" * 100)
    print()

    print(f"{'Condition':25s} {'TP':>5} {'SL':>5} {'SL/TP':>6} | {'Trades':>6} {'WR':>6} {'AvgPnL':>8} {'Total':>9} "
          f"{'PF':>5} {'MaxDD':>8} {'AvgHold':>8}")
    print("─" * 100)

    best_dual_ev = -999
    best_dual_config = None

    for cond_name in ["S5_vol_accel", "S6_chaos", "S2_quiet_storm"]:
        cond_func = conditions[cond_name]

        for tp in [20, 30, 40, 50, 75]:
            for sl_mult in [1, 1.5, 2, 3, 5]:
                sl = round(tp * sl_mult)

                trades = simulate_dual(closes, states, cond_func, tp, sl, max_hold,
                                       cooldown=5, start_idx=split)
                if len(trades) < 5:
                    continue

                r = analyze_trades(trades, f"DUAL {cond_name}")
                if r is None:
                    continue

                if r['total_pnl'] > 0 or r['wr'] >= 0.50:
                    pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "INF"
                    marker = " +EV" if r['avg_pnl'] > 0 else ""
                    print(f"{cond_name:25s} ${tp:>4} ${sl:>4} {sl_mult:>5.1f}× | "
                          f"{r['n']:>6} {r['wr']:>5.1%} ${r['avg_pnl']:>+7.1f} ${r['total_pnl']:>+8.0f} "
                          f"{pf_str:>5} ${r['max_dd']:>7.0f} {r['med_hold']:>7.0f}m{marker}")

                    if r['avg_pnl'] > best_dual_ev and r['n'] >= 10:
                        best_dual_ev = r['avg_pnl']
                        best_dual_config = (cond_name, tp, sl, sl_mult, r)

    if best_dual_config:
        name, tp, sl, mult, r = best_dual_config
        print(f"\n  ★ BEST DUAL: {name} TP=${tp} SL=${sl} ({mult}×)")
        print(f"    {r['n']} trades | WR={r['wr']:.1%} | EV=${r['avg_pnl']:+.1f} | Total=${r['total_pnl']:+,.0f}")

    # ══════════════════════════════════════════════════════
    # PART 3: SPEED ANALYSIS — How fast are trades?
    # ══════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("PART 3: SPEED — Average trade duration")
    print("=" * 100)
    print()

    for cond_name in ["S1_ultra_quiet", "S2_quiet_storm", "S3_quiet_decouple"]:
        cond_func = conditions[cond_name]
        for tp in [30, 45, 60]:
            sl = tp * 3  # Fixed 3× SL for speed comparison
            trades = simulate_sequential(closes, states, cond_func, 'LONG', tp, sl, max_hold,
                                         cooldown=3, start_idx=split)
            if len(trades) < 3:
                continue

            hold_times = [t['hold_time'] for t in trades]
            tp_times = [t['hold_time'] for t in trades if t['reason'] == 'TP']
            sl_times = [t['hold_time'] for t in trades if t['reason'] == 'SL']
            wr = sum(1 for t in trades if t['reason'] == 'TP') / len(trades)

            print(f"  {cond_name:25s} TP=${tp} SL=${sl}")
            print(f"    {len(trades)} trades | WR={wr:.1%}")
            print(f"    All:  avg={np.mean(hold_times):.0f}m med={np.median(hold_times):.0f}m")
            if tp_times:
                print(f"    Wins: avg={np.mean(tp_times):.0f}m med={np.median(tp_times):.0f}m max={max(tp_times)}m")
            if sl_times:
                print(f"    Loss: avg={np.mean(sl_times):.0f}m med={np.median(sl_times):.0f}m")
            print()

    # ══════════════════════════════════════════════════════
    # PART 4: EQUITY CURVE — Best config
    # ══════════════════════════════════════════════════════
    if best_config:
        name, tp, sl, mult, _ = best_config
        print("=" * 100)
        print(f"PART 4: EQUITY CURVE — {name} TP=${tp} SL=${sl}")
        print("=" * 100)
        print()

        # Run on FULL data
        trades = simulate_sequential(closes, states, conditions[name], 'LONG', tp, sl, max_hold, cooldown=5)
        pnls = [t['pnl'] for t in trades]
        equity = np.cumsum(pnls)

        print(f"  Full period: {len(trades)} trades")
        print(f"  Final equity: ${equity[-1]:+,.0f}" if len(equity) > 0 else "  No trades")

        # Print trade-by-trade
        print(f"\n  {'#':>4} {'Bar':>6} {'Side':>5} {'Entry':>10} {'Exit':>10} {'PnL':>8} {'Reason':>8} {'Hold':>5} {'Equity':>10}")
        print(f"  {'─'*75}")

        cum = 0
        for i, t in enumerate(trades):
            cum += t['pnl']
            print(f"  {i+1:>4} {t['entry_bar']:>6} {t['side']:>5} ${t['entry_price']:>9,.0f} "
                  f"${t['exit_price']:>9,.0f} ${t['pnl']:>+7.0f} {t['reason']:>8} {t['hold_time']:>4}m ${cum:>+9,.0f}")

    # ══════════════════════════════════════════════════════
    # PART 5: MINIMUM SIZE CALCULATION
    # ══════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("PART 5: MINIMUM SIZE & CAPITAL")
    print("=" * 100)
    print()
    print("  Hyperliquid BTC-PERP:")
    print("    Min order: 0.001 BTC ≈ $80")
    print("    Max leverage: 50×")
    print()

    if best_config:
        name, tp, sl, mult, r = best_config
        print(f"  Best config: TP=${tp} SL=${sl}")
        print()

        for size_btc in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
            size_usd = size_btc * 80000
            tp_usd = tp * size_btc
            sl_usd = sl * size_btc
            ev_usd = r['avg_pnl'] * size_btc
            max_dd_usd = abs(r['max_dd']) * size_btc

            # Required margin at various leverages
            for lev in [10, 20, 50]:
                margin = size_usd / lev
                print(f"    {size_btc:.3f} BTC (${size_usd:>8,.0f} notional) @ {lev:>2}× "
                      f"→ margin=${margin:>7,.0f} | TP=${tp_usd:>+6.2f} SL=${sl_usd:>+7.2f} "
                      f"EV=${ev_usd:>+6.2f}/trade | MaxDD=${max_dd_usd:>7.2f}")
            print()


if __name__ == "__main__":
    main()
