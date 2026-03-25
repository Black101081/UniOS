"""
Sniper 33-dim Strategy — NautilusTrader wrapper for the state-vector edge approach.

Uses a 33-dimensional market state vector to find high-probability entries:
  - 6 base metrics: vol, autocorr, skew, kurt, vol_anom, pv_corr
  - 6 first derivatives (d_*)
  - 6 second derivatives (dd_*)
  - 15 cross-correlations (xc_*)

Entry logic:
  - "Quiet market" conditions (low vol + specific cross-correlation regimes)
  - Direction decided by skew + autocorrelation (AUTO mode)
  - Fixed TP/SL in price terms, max holding period

Sequential: only 1 position at a time with cooldown between trades.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


SCALES = [5, 10, 20, 50, 100]
DIMS = ["vol", "autocorr", "skew", "kurt", "vol_anom", "pv_corr"]


class SniperStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for Sniper 33-dim strategy."""

    bar_type: str
    instrument_id: str

    # Position sizing
    position_size: float = 0.01  # BTC

    # TP/SL in price terms (USD for BTC)
    tp_price: float = 50.0
    sl_price: float = 150.0

    # Trade management
    max_hold_bars: int = 120        # max bars before timeout close
    cooldown_bars: int = 5          # bars between trades

    # State vector warmup
    warmup_bars: int = 220          # need ~200 bars for metrics + 20 for percentiles

    # Condition: which entry filter to use
    # "ultra", "storm", "decouple", "quiet", "relaxed"
    condition: str = "storm"

    # Direction: "LONG", "SHORT", or "AUTO"
    direction: str = "AUTO"

    # Percentile thresholds for adaptive conditions
    vol_percentile: float = 10.0     # vol <= Pxx
    xc_sp_percentile: float = 15.0   # xc_skew_pv_corr <= Pxx
    xc_va_percentile: float = 25.0   # xc_vol_autocorr <= Pxx

    # Calibration window (how many states to collect before trading)
    calibration_bars: int = 2000

    # Gap detection
    gap_seconds: int = 300


class SniperStrategy(Strategy):
    """
    33-dimensional state vector sniper strategy for NautilusTrader.

    Accumulates bar data, computes the state vector internally,
    calibrates percentile thresholds, then trades.
    """

    def __init__(self, config: SniperStrategyConfig) -> None:
        super().__init__(config)

        self.bar_type = BarType.from_str(config.bar_type)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.instrument: Instrument | None = None

        # Bar history for state computation (bounded rolling windows)
        self._closes: deque[float] = deque(maxlen=250)
        self._volumes: deque[float] = deque(maxlen=250)
        self._returns: deque[float] = deque(maxlen=250)

        # Metric history (for derivatives — only need last 25)
        self._metric_history: deque[dict] = deque(maxlen=25)

        # Calibration: only store vol/xc values, not full states
        self._cal_vol: list[float] = []
        self._cal_xc_sp: list[float] = []
        self._cal_xc_va: list[float] = []
        self._cal_count: int = 0

        # Calibration thresholds (computed after calibration_bars)
        self._calibrated = False
        self._vol_threshold: float = 0.0
        self._xc_sp_threshold: float = 0.0
        self._xc_va_threshold: float = 0.0

        # Position tracking
        self._bar_count: int = 0
        self._prev_bar_ts: int = 0
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._entry_side: OrderSide | None = None
        self._bars_since_entry: int = 0
        self._cooldown_remaining: int = 0

        # Stats
        self.stats = {
            "total_bars": 0,
            "states_computed": 0,
            "conditions_triggered": 0,
            "entries_long": 0,
            "entries_short": 0,
            "exits_tp": 0,
            "exits_sl": 0,
            "exits_timeout": 0,
            "exits_gap": 0,
            "total_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "gaps_detected": 0,
        }

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found")
            return

        self.subscribe_bars(self.bar_type)
        self.log.info("SniperStrategy (33-dim state vector) started")

    def on_bar(self, bar: Bar) -> None:
        self._bar_count += 1
        self.stats["total_bars"] = self._bar_count

        close = float(bar.close)
        volume = float(bar.volume)

        # Gap detection
        if self._prev_bar_ts > 0:
            gap_s = (bar.ts_event - self._prev_bar_ts) / 1_000_000_000
            if gap_s > self.config.gap_seconds:
                self._on_gap()
        self._prev_bar_ts = bar.ts_event

        # Accumulate
        self._closes.append(close)
        self._volumes.append(volume)
        if len(self._closes) >= 2:
            prev_close = self._closes[-2]
            ret = (close - prev_close) / prev_close if prev_close > 0 else 0.0
            self._returns.append(ret)
        else:
            self._returns.append(0.0)

        # Need enough bars for state computation
        if len(self._closes) < self.config.warmup_bars:
            return

        # Compute state vector
        state = self._compute_state()
        if state is None:
            return

        self.stats["states_computed"] += 1

        # Calibrate thresholds once we have enough states
        if not self._calibrated:
            self._cal_vol.append(state["vol"])
            self._cal_xc_sp.append(state.get("xc_skew_pv_corr", 0))
            self._cal_xc_va.append(state.get("xc_vol_autocorr", 0))
            self._cal_count += 1
            if self._cal_count >= self.config.calibration_bars:
                self._calibrate()
            else:
                return

        # Position management
        if self._in_position:
            self._bars_since_entry += 1
            self._manage_position(close)
            return

        # Cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return

        # Check entry condition
        if self._check_condition(state):
            self.stats["conditions_triggered"] += 1
            side = self._determine_direction(state)
            if side is not None:
                self._enter(side, close)

    # ─────────────────────────────────────────────────────────────
    # State vector computation (33 dimensions)
    # ─────────────────────────────────────────────────────────────
    def _compute_state(self) -> dict | None:
        """Compute the 33-dimensional state vector from recent bars."""
        n = len(self._closes)
        returns = np.array(list(self._returns)[-201:], dtype=np.float64)
        volumes = np.array(list(self._volumes)[-201:], dtype=np.float64)

        if len(returns) < 100:
            return None

        metrics = self._compute_metrics(returns, volumes)
        self._metric_history.append(metrics)

        if len(self._metric_history) < 3:
            return None

        cur = self._metric_history[-1]
        prev = self._metric_history[-2]
        prev2 = self._metric_history[-3]

        state = {}

        # Base + first derivative + second derivative = 18 dims
        for nm in DIMS:
            state[nm] = cur[nm]
            state[f"d_{nm}"] = cur[nm] - prev[nm]
            state[f"dd_{nm}"] = (cur[nm] - prev[nm]) - (prev[nm] - prev2[nm])

        # Cross-correlations = 15 dims
        if len(self._metric_history) >= 12:
            start = max(0, len(self._metric_history) - 20)
            vels = []
            for j in range(start, len(self._metric_history) - 1):
                vels.append({
                    nm: self._metric_history[j + 1][nm] - self._metric_history[j][nm]
                    for nm in DIMS
                })

            if len(vels) >= 5:
                for ii, n1 in enumerate(DIMS):
                    for n2 in DIMS[ii + 1:]:
                        v1 = np.array([v[n1] for v in vels])
                        v2 = np.array([v[n2] for v in vels])
                        if np.std(v1) > 1e-12 and np.std(v2) > 1e-12:
                            c = float(np.corrcoef(v1, v2)[0, 1])
                            state[f"xc_{n1}_{n2}"] = 0.0 if np.isnan(c) else c
                        else:
                            state[f"xc_{n1}_{n2}"] = 0.0
            else:
                self._fill_zero_xc(state)
        else:
            self._fill_zero_xc(state)

        return state

    @staticmethod
    def _fill_zero_xc(state: dict) -> None:
        for ii, n1 in enumerate(DIMS):
            for n2 in DIMS[ii + 1:]:
                state[f"xc_{n1}_{n2}"] = 0.0

    @staticmethod
    def _compute_metrics(returns: np.ndarray, volumes: np.ndarray) -> dict:
        """Compute 6 base metrics across multiple scales."""
        def _ms(func):
            vals = [func(s) for s in SCALES if len(returns) >= s]
            return float(np.mean(vals)) if vals else 0.0

        def _vol(s):
            return float(np.std(returns[-s:]))

        def _ac(s):
            r = returns[-s:]
            if np.std(r) > 1e-12:
                return float(np.corrcoef(r[1:], r[:-1])[0, 1])
            return 0.0

        def _sk(s):
            r = returns[-s:]
            std = np.std(r)
            if std > 1e-12:
                return float(np.mean(((r - r.mean()) / std) ** 3))
            return 0.0

        def _ku(s):
            r = returns[-s:]
            std = np.std(r)
            if std > 1e-12:
                return float(np.mean(((r - r.mean()) / std) ** 4))
            return 0.0

        def _va(s):
            w = volumes[-s:]
            med = np.median(w)
            return float(volumes[-1] / med) if med > 1e-12 else 1.0

        def _pv(s):
            r = np.abs(returns[-s:])
            v = volumes[-s:]
            if np.std(r) > 1e-12 and np.std(v) > 1e-12:
                return float(np.corrcoef(r, v)[0, 1])
            return 0.0

        return {
            "vol": _ms(_vol),
            "autocorr": _ms(_ac),
            "skew": _ms(_sk),
            "kurt": _ms(_ku),
            "vol_anom": _ms(_va),
            "pv_corr": _ms(_pv),
        }

    # ─────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────
    def _calibrate(self) -> None:
        """Compute percentile thresholds from collected calibration values."""
        self._vol_threshold = float(np.percentile(self._cal_vol, self.config.vol_percentile))
        self._xc_sp_threshold = float(np.percentile(self._cal_xc_sp, self.config.xc_sp_percentile))
        self._xc_va_threshold = float(np.percentile(self._cal_xc_va, self.config.xc_va_percentile))

        self._calibrated = True

        # Free calibration memory
        self._cal_vol.clear()
        self._cal_xc_sp.clear()
        self._cal_xc_va.clear()

        self.log.info(
            f"CALIBRATED after {self._cal_count} states: "
            f"vol_thr={self._vol_threshold:.6f} "
            f"xc_sp_thr={self._xc_sp_threshold:.4f} "
            f"xc_va_thr={self._xc_va_threshold:.4f}"
        )

    # ─────────────────────────────────────────────────────────────
    # Entry conditions
    # ─────────────────────────────────────────────────────────────
    def _check_condition(self, state: dict) -> bool:
        """Check if the state meets the entry condition."""
        cond = self.config.condition
        vol = state["vol"]
        xc_sp = state.get("xc_skew_pv_corr", 0)
        xc_va = state.get("xc_vol_autocorr", 0)

        # Use a tighter vol threshold for ultra/storm
        vol_p5 = self._vol_threshold * 0.5  # approximate P5 from P10

        if cond == "ultra":
            return vol <= vol_p5 and xc_sp <= self._xc_sp_threshold * 0.67
        elif cond == "storm":
            return vol <= self._vol_threshold and xc_sp <= self._xc_sp_threshold
        elif cond == "decouple":
            return vol <= self._vol_threshold and xc_va <= self._xc_va_threshold
        elif cond == "quiet":
            # Slightly relaxed vol threshold (P15 ~ 1.5× P10)
            return vol <= self._vol_threshold * 1.5
        elif cond == "relaxed":
            return vol <= self._vol_threshold * 2.0
        else:
            self.log.warning(f"Unknown condition: {cond}, defaulting to storm")
            return vol <= self._vol_threshold and xc_sp <= self._xc_sp_threshold

    def _determine_direction(self, state: dict) -> OrderSide | None:
        """Determine trade direction from state vector."""
        direction = self.config.direction

        if direction == "LONG":
            return OrderSide.BUY
        elif direction == "SHORT":
            return OrderSide.SELL
        elif direction == "AUTO":
            sk = state.get("skew", 0)
            ac = state.get("autocorr", 0)
            if sk > 0 or ac > 0.02:
                return OrderSide.BUY
            elif sk < 0 or ac < -0.02:
                return OrderSide.SELL
            return None
        return None

    # ─────────────────────────────────────────────────────────────
    # Order execution
    # ─────────────────────────────────────────────────────────────
    def _enter(self, side: OrderSide, price: float) -> None:
        """Submit a bracket order."""
        tp_dist = self.config.tp_price
        sl_dist = self.config.sl_price

        if side == OrderSide.BUY:
            sl_price = price - sl_dist
            tp_price = price + tp_dist
        else:
            sl_price = price + sl_dist
            tp_price = price - tp_dist

        sl_price_obj = self.instrument.make_price(sl_price)
        tp_price_obj = self.instrument.make_price(tp_price)
        qty = self.instrument.make_qty(self.config.position_size)

        bracket = self.order_factory.bracket(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=qty,
            sl_trigger_price=sl_price_obj,
            tp_price=tp_price_obj,
        )
        self.submit_order_list(bracket)

        self._in_position = True
        self._entry_price = price
        self._entry_side = side
        self._bars_since_entry = 0

        if side == OrderSide.BUY:
            self.stats["entries_long"] += 1
        else:
            self.stats["entries_short"] += 1

        self.log.info(
            f"SNIPER ENTRY: {side.name} @ {price:.1f} | "
            f"TP={float(tp_price_obj):.1f} SL={float(sl_price_obj):.1f} | "
            f"bar={self._bar_count}"
        )

    # ─────────────────────────────────────────────────────────────
    # Position management
    # ─────────────────────────────────────────────────────────────
    def _manage_position(self, price: float) -> None:
        """Check timeout exit (TP/SL handled by bracket orders)."""
        if self._bars_since_entry >= self.config.max_hold_bars:
            self._close_position("TIMEOUT", price)

    def _close_position(self, reason: str, price: float) -> None:
        """Close position and cancel remaining bracket orders."""
        pnl = self._calc_pnl(price)

        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

        self.stats[f"exits_{reason.lower()}"] = self.stats.get(f"exits_{reason.lower()}", 0) + 1
        self.stats["total_pnl"] += pnl
        if pnl > 0:
            self.stats["wins"] += 1
        elif pnl < 0:
            self.stats["losses"] += 1

        self.log.info(
            f"SNIPER EXIT: {reason} @ {price:.1f} | PnL={pnl:+.1f} | "
            f"hold={self._bars_since_entry} bars"
        )
        self._reset_position()

    def _calc_pnl(self, price: float) -> float:
        if self._entry_side == OrderSide.BUY:
            return price - self._entry_price
        else:
            return self._entry_price - price

    def _reset_position(self) -> None:
        self._in_position = False
        self._entry_price = 0.0
        self._entry_side = None
        self._bars_since_entry = 0
        self._cooldown_remaining = self.config.cooldown_bars

    # ─────────────────────────────────────────────────────────────
    # Order events
    # ─────────────────────────────────────────────────────────────
    def on_order_filled(self, event) -> None:
        if self.portfolio.is_flat(self.instrument_id) and self._in_position:
            order = self.cache.order(event.client_order_id)
            fill_price = float(event.last_px)
            pnl = self._calc_pnl(fill_price)

            if order and order.order_type.name == "STOP_MARKET":
                self.stats["exits_sl"] += 1
                self.log.info(f"SNIPER EXIT: SL @ {fill_price:.1f} | PnL={pnl:+.1f}")
            elif order and order.order_type.name == "LIMIT" and order.is_reduce_only:
                self.stats["exits_tp"] += 1
                self.log.info(f"SNIPER EXIT: TP @ {fill_price:.1f} | PnL={pnl:+.1f}")

            self.stats["total_pnl"] += pnl
            if pnl > 0:
                self.stats["wins"] += 1
            elif pnl < 0:
                self.stats["losses"] += 1

            self._reset_position()

    # ─────────────────────────────────────────────────────────────
    # Gap handling
    # ─────────────────────────────────────────────────────────────
    def _on_gap(self) -> None:
        self.stats["gaps_detected"] += 1
        self.log.info("Gap detected — resetting state")

        if self._in_position:
            self.cancel_all_orders(self.instrument_id)
            self.close_all_positions(self.instrument_id)
            self.stats["exits_gap"] += 1
            self._reset_position()

        # Reset metric history but keep price/volume (deques auto-bounded)
        self._metric_history.clear()

    # ─────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────
    def on_stop(self) -> None:
        s = self.stats
        total_trades = s["entries_long"] + s["entries_short"]
        win_rate = s["wins"] / total_trades * 100 if total_trades > 0 else 0

        self.log.info("=" * 60)
        self.log.info("SNIPER 33-DIM STRATEGY — FINAL STATISTICS")
        self.log.info("=" * 60)
        self.log.info(f"Total bars processed: {s['total_bars']}")
        self.log.info(f"States computed: {s['states_computed']}")
        self.log.info(f"Calibrated: {self._calibrated}")
        self.log.info(f"Condition triggers: {s['conditions_triggered']}")
        self.log.info(f"Gaps detected: {s['gaps_detected']}")
        self.log.info(f"--- Entries ---")
        self.log.info(f"  LONG:  {s['entries_long']}")
        self.log.info(f"  SHORT: {s['entries_short']}")
        self.log.info(f"  Total: {total_trades}")
        self.log.info(f"--- Exits ---")
        self.log.info(f"  TP:      {s['exits_tp']}")
        self.log.info(f"  SL:      {s['exits_sl']}")
        self.log.info(f"  Timeout: {s['exits_timeout']}")
        self.log.info(f"  Gap:     {s['exits_gap']}")
        self.log.info(f"--- Performance ---")
        self.log.info(f"  Wins:     {s['wins']}")
        self.log.info(f"  Losses:   {s['losses']}")
        self.log.info(f"  Win rate: {win_rate:.1f}%")
        self.log.info(f"  Total PnL: {s['total_pnl']:.2f}")
        self.log.info("=" * 60)
