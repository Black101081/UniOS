#!/usr/bin/env python3
"""
Phân tích chất lượng trade thực tế
===================================
Cho mỗi signal trigger:
1. Giá tăng bao nhiêu $ trước khi quay đầu (Max Favorable Excursion)
2. Giá giảm bao nhiêu $ trước khi hit TP (Max Adverse Excursion = drawdown)
3. Mất bao lâu để hit TP
4. Với SL ở các mức khác nhau → tỷ lệ win thay đổi thế nào
5. R:R ratio thực tế
"""

import numpy as np
import pandas as pd
from pathlib import Path
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
    return states


def condition_p1(state):
    """P1_quiet_storm: vol <= 0.00017 AND xc_skew_pv_corr <= -0.49"""
    return state.get("vol", 999) <= 0.00017 and state.get("xc_skew_pv_corr", 0) <= -0.49

def condition_p5(state):
    """P5_quiet_no_confirm: vol <= 0.00017 AND pv_corr <= 0.034"""
    return state.get("vol", 999) <= 0.00017 and state.get("pv_corr", 999) <= 0.034


def analyze_trades(closes, states, condition_func, condition_name, max_hold=360):
    """
    Cho mỗi bar trigger condition:
    - Track giá từ entry đến max_hold bars sau
    - Đo MFE, MAE, time to TP, time to SL
    """
    n = len(closes)
    trades = []

    for i in range(n):
        if states[i] is None:
            continue
        if not condition_func(states[i]):
            continue

        entry_price = closes[i]
        end = min(i + max_hold, n)
        future = closes[i+1:end] if i+1 < end else np.array([])

        if len(future) == 0:
            continue

        # Price path relative to entry
        pnl_path = future - entry_price  # for BUY

        # Max Favorable Excursion (max profit before closing)
        mfe = float(np.max(pnl_path)) if len(pnl_path) > 0 else 0
        # Max Adverse Excursion (max drawdown before closing)
        mae = float(np.min(pnl_path)) if len(pnl_path) > 0 else 0

        # Time to various TP levels
        tp_times = {}
        for tp in [10, 20, 30, 40, 45, 50, 60, 75, 100, 125, 150, 200]:
            hit_idx = np.where(pnl_path >= tp)[0]
            if len(hit_idx) > 0:
                tp_times[tp] = int(hit_idx[0]) + 1  # bars to hit
            else:
                tp_times[tp] = -1  # never hit

        # Time to various SL levels (negative = price went down)
        sl_times = {}
        for sl in [10, 20, 30, 40, 50, 75, 100, 150, 200, 300]:
            hit_idx = np.where(pnl_path <= -sl)[0]
            if len(hit_idx) > 0:
                sl_times[sl] = int(hit_idx[0]) + 1
            else:
                sl_times[sl] = -1

        # Drawdown before hitting each TP
        dd_before_tp = {}
        for tp in [40, 45, 50, 60, 75, 100]:
            tp_idx = tp_times.get(tp, -1)
            if tp_idx > 0:
                dd_before_tp[tp] = float(np.min(pnl_path[:tp_idx]))
            else:
                dd_before_tp[tp] = float(mae)  # never hit TP, use overall MAE

        trades.append({
            "bar_idx": i,
            "entry_price": entry_price,
            "mfe": mfe,
            "mae": mae,
            "tp_times": tp_times,
            "sl_times": sl_times,
            "dd_before_tp": dd_before_tp,
            "future_bars": len(future),
        })

    return trades


def print_analysis(trades, condition_name):
    if not trades:
        print(f"  No trades for {condition_name}")
        return

    n = len(trades)
    print(f"\n{'='*80}")
    print(f"CONDITION: {condition_name} | {n} trades")
    print(f"{'='*80}")

    # MFE / MAE distribution
    mfes = [t["mfe"] for t in trades]
    maes = [t["mae"] for t in trades]
    print(f"\n  Max Favorable Excursion (max profit before any exit):")
    print(f"    Min: ${min(mfes):.0f} | P25: ${np.percentile(mfes, 25):.0f} | "
          f"Median: ${np.percentile(mfes, 50):.0f} | P75: ${np.percentile(mfes, 75):.0f} | "
          f"Max: ${max(mfes):.0f}")

    print(f"\n  Max Adverse Excursion (max drawdown before any exit):")
    print(f"    Min: ${min(maes):.0f} | P25: ${np.percentile(maes, 25):.0f} | "
          f"Median: ${np.percentile(maes, 50):.0f} | P75: ${np.percentile(maes, 75):.0f} | "
          f"Max: ${max(maes):.0f}")

    # TP hit rate at various levels
    print(f"\n  TP Hit Rate (BUY direction, max hold 6h):")
    print(f"  {'TP':>6} {'Hit%':>7} {'Avg bars':>10} {'Med bars':>10} {'Avg DD':>10} {'Max DD':>10}")
    print(f"  {'─'*60}")
    for tp in [10, 20, 30, 40, 45, 50, 60, 75, 100, 125, 150, 200]:
        hit_count = sum(1 for t in trades if t["tp_times"].get(tp, -1) > 0)
        hit_pct = hit_count / n
        hit_bars = [t["tp_times"][tp] for t in trades if t["tp_times"].get(tp, -1) > 0]
        avg_bars = np.mean(hit_bars) if hit_bars else 0
        med_bars = np.median(hit_bars) if hit_bars else 0

        # Average drawdown before hitting this TP
        dds = [t["dd_before_tp"].get(tp, t["mae"]) for t in trades if t["tp_times"].get(tp, -1) > 0]
        avg_dd = np.mean(dds) if dds else 0
        max_dd = min(dds) if dds else 0

        print(f"  ${tp:>5} {hit_pct:>6.1%} {avg_bars:>9.0f}m {med_bars:>9.0f}m "
              f"${avg_dd:>9.0f} ${max_dd:>9.0f}")

    # SL hit rate
    print(f"\n  SL Hit Rate (how often does price drop to -$X WITHIN 6h):")
    print(f"  {'SL':>6} {'Hit%':>7} {'Avg bars':>10}")
    print(f"  {'─'*30}")
    for sl in [10, 20, 30, 40, 50, 75, 100, 150, 200, 300]:
        hit_count = sum(1 for t in trades if t["sl_times"].get(sl, -1) > 0)
        hit_pct = hit_count / n
        hit_bars = [t["sl_times"][sl] for t in trades if t["sl_times"].get(sl, -1) > 0]
        avg_bars = np.mean(hit_bars) if hit_bars else 0
        print(f"  ${sl:>5} {hit_pct:>6.1%} {avg_bars:>9.0f}m")

    # TP/SL combinations: win rate with bracket orders
    print(f"\n  BRACKET ORDERS — TP + SL combos (ai đến trước?):")
    print(f"  {'TP':>5} {'SL':>5} {'Win%':>7} {'Wins':>5} {'Losses':>7} {'Neither':>8} {'Avg_W_time':>11} {'Avg_L_time':>11} {'R:R':>6} {'EV/trade':>9}")
    print(f"  {'─'*85}")

    for tp in [30, 40, 45, 50, 60, 75, 100]:
        for sl in [20, 30, 40, 50, 75, 100, 150]:
            wins = 0
            losses = 0
            neither = 0
            win_times = []
            loss_times = []

            for t in trades:
                tp_bar = t["tp_times"].get(tp, -1)
                sl_bar = t["sl_times"].get(sl, -1)

                if tp_bar > 0 and sl_bar > 0:
                    if tp_bar <= sl_bar:
                        wins += 1
                        win_times.append(tp_bar)
                    else:
                        losses += 1
                        loss_times.append(sl_bar)
                elif tp_bar > 0:
                    wins += 1
                    win_times.append(tp_bar)
                elif sl_bar > 0:
                    losses += 1
                    loss_times.append(sl_bar)
                else:
                    neither += 1

            total_decided = wins + losses
            if total_decided < 5:
                continue

            win_pct = wins / total_decided
            avg_w_time = np.mean(win_times) if win_times else 0
            avg_l_time = np.mean(loss_times) if loss_times else 0
            rr = tp / sl
            ev = win_pct * tp - (1 - win_pct) * sl  # expected value per trade

            marker = ""
            if win_pct >= 0.90:
                marker = " ★★★"
            elif win_pct >= 0.80:
                marker = " ★★"
            elif win_pct >= 0.70:
                marker = " ★"
            if ev > 0:
                marker += " +EV"

            print(f"  ${tp:>4} ${sl:>4} {win_pct:>6.1%} {wins:>5} {losses:>7} {neither:>8} "
                  f"{avg_w_time:>10.0f}m {avg_l_time:>10.0f}m {rr:>5.1f}R "
                  f"${ev:>+7.1f}{marker}")

    # Best configurations
    print(f"\n  TOP 5 CONFIGURATIONS (by Expected Value):")
    configs = []
    for tp in [30, 40, 45, 50, 60, 75, 100]:
        for sl in [20, 30, 40, 50, 75, 100, 150]:
            wins = losses = 0
            for t in trades:
                tp_bar = t["tp_times"].get(tp, -1)
                sl_bar = t["sl_times"].get(sl, -1)
                if tp_bar > 0 and sl_bar > 0:
                    if tp_bar <= sl_bar: wins += 1
                    else: losses += 1
                elif tp_bar > 0: wins += 1
                elif sl_bar > 0: losses += 1
            total = wins + losses
            if total < 5:
                continue
            wr = wins / total
            ev = wr * tp - (1 - wr) * sl
            configs.append((tp, sl, wr, wins, losses, ev, total))

    configs.sort(key=lambda x: x[5], reverse=True)
    for i, (tp, sl, wr, w, l, ev, tot) in enumerate(configs[:5]):
        print(f"    #{i+1}: TP=${tp} SL=${sl} → Win={wr:.1%} ({w}W/{l}L) "
              f"R:R={tp/sl:.1f} EV=${ev:+.1f}/trade")

    # Worst case analysis
    print(f"\n  WORST CASE (trades that went against us):")
    losing_trades = [t for t in trades if t["mae"] < -30]
    if losing_trades:
        print(f"    Trades with >$30 drawdown: {len(losing_trades)}/{n} ({len(losing_trades)/n:.1%})")
        for lt in sorted(losing_trades, key=lambda x: x["mae"])[:5]:
            print(f"      Entry ${lt['entry_price']:,.0f} | MAE ${lt['mae']:,.0f} | MFE ${lt['mfe']:,.0f}")
    else:
        print(f"    No trades with >$30 drawdown!")

    worst_mae = min(t["mae"] for t in trades)
    print(f"    Worst single drawdown: ${worst_mae:,.0f}")


def main():
    data_path = Path(__file__).parent.parent / "data" / "btc_candles_1min.parquet"
    df = pd.read_parquet(data_path).sort_values("timestamp").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)

    print(f"Data: {n:,} candles | ${closes.min():,.0f} — ${closes.max():,.0f}")
    print()

    print("Building state vectors...")
    t0 = time.time()
    states = build_states(closes, volumes)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Analyze P1_quiet_storm
    trades_p1 = analyze_trades(closes, states, condition_p1, "P1_quiet_storm", max_hold=360)
    print_analysis(trades_p1, "P1_quiet_storm (vol<=0.00017 & xc_skew_pv_corr<=-0.49)")

    # Analyze P5_quiet_no_confirm
    trades_p5 = analyze_trades(closes, states, condition_p5, "P5_quiet_no_confirm", max_hold=360)
    print_analysis(trades_p5, "P5_quiet_no_confirm (vol<=0.00017 & pv_corr<=0.034)")


if __name__ == "__main__":
    main()
