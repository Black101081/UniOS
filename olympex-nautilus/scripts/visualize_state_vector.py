#!/usr/bin/env python3
"""Visualize the 33-dimensional state vector architecture."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from itertools import combinations

# ── colours ──────────────────────────────────────────────────────────────
BG       = "#0d1117"
CARD_BG  = "#161b22"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
PINK     = "#f778ba"
RED      = "#f85149"
WHITE    = "#e6edf3"
GREY     = "#8b949e"
CYAN     = "#39d353"

LAYER_COLORS = [ACCENT, GREEN, ORANGE, PURPLE]
METRIC_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#f778ba"]

METRICS = ["vol", "autocorr", "skew", "kurt", "vol_anom", "pv_corr"]
METRIC_LABELS = ["Volatility", "Autocorrelation", "Skewness", "Kurtosis", "Volume\nAnomaly", "Price-Volume\nCorrelation"]
SCALES = [5, 10, 20, 50, 100]

fig = plt.figure(figsize=(24, 32), facecolor=BG)

# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 1: Title & Overview  (top)                         ║
# ╚══════════════════════════════════════════════════════════════╝
ax_title = fig.add_axes([0, 0.92, 1, 0.08], facecolor=BG)
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
ax_title.axis("off")

ax_title.text(0.5, 0.72, "SNIPER 33-DIMENSIONAL STATE VECTOR",
              fontsize=32, fontweight="bold", color=WHITE, ha="center", va="center",
              fontfamily="monospace")
ax_title.text(0.5, 0.35, "6 Base Metrics  ×  4 Layers  =  6 + 6 + 6 + C(6,2)  =  33 dimensions",
              fontsize=16, color=GREY, ha="center", va="center", fontfamily="monospace")
ax_title.text(0.5, 0.10, "Multi-scale consensus across M1 candles  |  Scales: 5, 10, 20, 50, 100",
              fontsize=13, color=GREY, ha="center", va="center", fontfamily="monospace")

# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 2: 6 Base Metrics with multi-scale (middle-top)    ║
# ╚══════════════════════════════════════════════════════════════╝
ax_metrics = fig.add_axes([0.03, 0.73, 0.94, 0.18], facecolor=BG)
ax_metrics.set_xlim(0, 12); ax_metrics.set_ylim(0, 3.5)
ax_metrics.axis("off")

ax_metrics.text(6, 3.3, "── 6 BASE METRICS (Multi-Scale Consensus) ──",
                fontsize=16, fontweight="bold", color=WHITE, ha="center",
                fontfamily="monospace")

for i, (m, ml) in enumerate(zip(METRICS, METRIC_LABELS)):
    x = 1 + i * 1.8
    col = METRIC_COLORS[i]

    # Main metric box
    rect = FancyBboxPatch((x - 0.7, 0.8), 1.4, 1.8, boxstyle="round,pad=0.1",
                           facecolor=CARD_BG, edgecolor=col, linewidth=2)
    ax_metrics.add_patch(rect)

    ax_metrics.text(x, 2.35, ml, fontsize=9, fontweight="bold", color=col,
                    ha="center", va="center", fontfamily="monospace")

    # Scale bars inside
    for j, s in enumerate(SCALES):
        bar_w = 0.15 + j * 0.08
        bar_y = 1.0 + j * 0.22
        ax_metrics.barh(bar_y, bar_w, height=0.14, left=x - 0.5,
                        color=col, alpha=0.3 + j * 0.15, edgecolor=col, linewidth=0.5)
        ax_metrics.text(x + 0.55, bar_y, f"s={s}", fontsize=6, color=GREY,
                        va="center", fontfamily="monospace")

    # Arrow down
    ax_metrics.annotate("", xy=(x, 0.65), xytext=(x, 0.8),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax_metrics.text(x, 0.45, f"dim {i}", fontsize=8, color=GREY, ha="center",
                    fontfamily="monospace")

# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 3: 4 Layers Breakdown (middle)                     ║
# ╚══════════════════════════════════════════════════════════════╝
ax_layers = fig.add_axes([0.03, 0.32, 0.94, 0.40], facecolor=BG)
ax_layers.set_xlim(0, 14); ax_layers.set_ylim(0, 10)
ax_layers.axis("off")

layers = [
    ("L0 — POSITION", "Current value", METRICS, 6, ACCENT,
     "state[m] = consensus(m, scales)"),
    ("L1 — VELOCITY", "1st derivative", [f"d_{m}" for m in METRICS], 6, GREEN,
     "d_m = state[m]ₜ − state[m]ₜ₋₁"),
    ("L2 — ACCELERATION", "2nd derivative", [f"dd_{m}" for m in METRICS], 6, ORANGE,
     "dd_m = d_mₜ − d_mₜ₋₁"),
]

xc_pairs = [f"xc_{a}_{b}" for a, b in combinations(METRICS, 2)]

y_positions = [9.0, 6.8, 4.6, 1.8]

for idx, (label, desc, dims, count, col, formula) in enumerate(layers):
    y = y_positions[idx]

    # Layer header box
    header = FancyBboxPatch((0.2, y - 0.4), 3.0, 0.8, boxstyle="round,pad=0.08",
                             facecolor=col, edgecolor=col, alpha=0.2, linewidth=2)
    ax_layers.add_patch(header)
    ax_layers.text(1.7, y + 0.05, label, fontsize=13, fontweight="bold", color=col,
                   ha="center", va="center", fontfamily="monospace")

    # Count badge
    badge = plt.Circle((3.7, y), 0.3, facecolor=col, edgecolor=col, alpha=0.3)
    ax_layers.add_patch(badge)
    ax_layers.text(3.7, y, str(count), fontsize=12, fontweight="bold", color=col,
                   ha="center", va="center", fontfamily="monospace")

    # Dimension boxes
    for j, d in enumerate(dims[:6]):
        bx = 4.5 + j * 1.5
        rect = FancyBboxPatch((bx - 0.55, y - 0.3), 1.1, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=CARD_BG, edgecolor=METRIC_COLORS[j % 6],
                               linewidth=1.5, alpha=0.8)
        ax_layers.add_patch(rect)
        ax_layers.text(bx, y, d, fontsize=7.5, color=METRIC_COLORS[j % 6],
                       ha="center", va="center", fontfamily="monospace",
                       fontweight="bold")

    # Formula
    ax_layers.text(13.5, y, formula, fontsize=9, color=GREY, ha="right",
                   va="center", fontfamily="monospace", style="italic")

    # Arrow between layers
    if idx < 2:
        ax_layers.annotate("", xy=(1.7, y - 0.6), xytext=(1.7, y - 0.4),
                           arrowprops=dict(arrowstyle="->", color=GREY, lw=1, ls="--"))

# ── Layer 3: Cross-correlation (special layout) ──
y = y_positions[3]
header = FancyBboxPatch((0.2, y - 0.4), 3.0, 0.8, boxstyle="round,pad=0.08",
                         facecolor=PURPLE, edgecolor=PURPLE, alpha=0.2, linewidth=2)
ax_layers.add_patch(header)
ax_layers.text(1.7, y + 0.05, "L3 — CROSS-CORR", fontsize=13, fontweight="bold",
               color=PURPLE, ha="center", va="center", fontfamily="monospace")

badge = plt.Circle((3.7, y), 0.3, facecolor=PURPLE, edgecolor=PURPLE, alpha=0.3)
ax_layers.add_patch(badge)
ax_layers.text(3.7, y, "15", fontsize=12, fontweight="bold", color=PURPLE,
               ha="center", va="center", fontfamily="monospace")

# Cross-corr grid: 3 rows x 5 cols
for j, xc in enumerate(xc_pairs):
    row = j // 5
    col_idx = j % 5
    bx = 4.5 + col_idx * 1.9
    by = y + 0.5 - row * 0.55

    # Colour by first metric in pair
    pair_idx = j
    c1 = METRIC_COLORS[0] if "vol_" in xc and not "anom" in xc.split("_")[1] else \
         METRIC_COLORS[1] if "autocorr" in xc.split("xc_")[1].split("_")[0] else \
         METRIC_COLORS[2] if "skew" in xc.split("xc_")[1].split("_")[0] else \
         METRIC_COLORS[3] if "kurt" in xc.split("xc_")[1].split("_")[0] else \
         METRIC_COLORS[4]

    rect = FancyBboxPatch((bx - 0.8, by - 0.2), 1.6, 0.4,
                           boxstyle="round,pad=0.04",
                           facecolor=CARD_BG, edgecolor=PURPLE, linewidth=1,
                           alpha=0.7)
    ax_layers.add_patch(rect)
    # Shorten label
    short = xc.replace("autocorr", "ac").replace("vol_anom", "va").replace("pv_corr", "pv")
    ax_layers.text(bx, by, short, fontsize=6.5, color=PURPLE,
                   ha="center", va="center", fontfamily="monospace")

ax_layers.text(13.5, y, "xc_A_B = corr(vel_A, vel_B)\nC(6,2) = 15 pairs",
               fontsize=9, color=GREY, ha="right", va="center",
               fontfamily="monospace", style="italic")

# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 4: Cross-Correlation Matrix (bottom-left)          ║
# ╚══════════════════════════════════════════════════════════════╝
ax_matrix = fig.add_axes([0.05, 0.04, 0.42, 0.27], facecolor=BG)

# Build 6x6 symmetric matrix
np.random.seed(42)
matrix = np.zeros((6, 6))
pair_list = list(combinations(range(6), 2))
for idx, (i, j) in enumerate(pair_list):
    val = np.random.uniform(-0.8, 0.8)
    matrix[i][j] = val
    matrix[j][i] = val
np.fill_diagonal(matrix, 1.0)

im = ax_matrix.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
short_names = ["vol", "ac", "skew", "kurt", "v_an", "pv"]
ax_matrix.set_xticks(range(6))
ax_matrix.set_yticks(range(6))
ax_matrix.set_xticklabels(short_names, fontsize=10, color=WHITE, fontfamily="monospace")
ax_matrix.set_yticklabels(short_names, fontsize=10, color=WHITE, fontfamily="monospace")
ax_matrix.tick_params(colors=GREY)

# Add correlation values
for i in range(6):
    for j in range(6):
        if i != j:
            ax_matrix.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center",
                          fontsize=8, color=WHITE, fontfamily="monospace",
                          fontweight="bold")

ax_matrix.set_title("Cross-Correlation Matrix (L3)\n15 unique pairs from C(6,2)",
                    fontsize=13, color=WHITE, fontfamily="monospace", pad=10)

cbar = plt.colorbar(im, ax=ax_matrix, shrink=0.8, aspect=20)
cbar.ax.tick_params(colors=GREY)
cbar.set_label("Correlation", color=GREY, fontsize=10)

# ╔══════════════════════════════════════════════════════════════╗
# ║  SECTION 5: Vector Assembly Diagram (bottom-right)          ║
# ╚══════════════════════════════════════════════════════════════╝
ax_vec = fig.add_axes([0.55, 0.04, 0.42, 0.27], facecolor=BG)
ax_vec.set_xlim(0, 10); ax_vec.set_ylim(0, 35)
ax_vec.axis("off")

ax_vec.text(5, 34.2, "33-DIM STATE VECTOR", fontsize=16, fontweight="bold",
            color=WHITE, ha="center", fontfamily="monospace")

# Draw the vector as stacked blocks
layer_info = [
    ("L0: Position", 6, ACCENT, METRICS),
    ("L1: Velocity", 6, GREEN, [f"d_{m}" for m in METRICS]),
    ("L2: Acceleration", 6, ORANGE, [f"dd_{m}" for m in METRICS]),
    ("L3: Cross-Corr", 15, PURPLE, xc_pairs),
]

y_cur = 32.5
for lname, count, col, dims in layer_info:
    # Layer label
    ax_vec.text(0.3, y_cur - count * 0.45 / 2, lname, fontsize=10, fontweight="bold",
                color=col, ha="left", va="center", fontfamily="monospace", rotation=0)

    for k in range(count):
        by = y_cur - k * 0.9
        # Dimension block
        rect = FancyBboxPatch((3.0, by - 0.35), 6.5, 0.7,
                               boxstyle="round,pad=0.03",
                               facecolor=col, edgecolor=col, alpha=0.15,
                               linewidth=1)
        ax_vec.add_patch(rect)

        # Index number
        dim_idx = sum(l[1] for l in layer_info[:layer_info.index((lname, count, col, dims))]) + k
        ax_vec.text(3.3, by, f"[{dim_idx:2d}]", fontsize=7, color=GREY,
                    va="center", fontfamily="monospace")

        # Dimension name
        dname = dims[k] if k < len(dims) else "..."
        short_d = dname.replace("autocorr", "ac").replace("vol_anom", "va").replace("pv_corr", "pv")
        ax_vec.text(5.0, by, short_d, fontsize=7.5, color=col,
                    va="center", fontfamily="monospace", fontweight="bold")

    y_cur -= count * 0.9 + 0.5

# Total badge at bottom
total_y = y_cur + 0.2
rect = FancyBboxPatch((3.0, total_y - 0.5), 6.5, 0.8,
                       boxstyle="round,pad=0.08",
                       facecolor=CYAN, edgecolor=CYAN, alpha=0.3, linewidth=2)
ax_vec.add_patch(rect)
ax_vec.text(6.25, total_y - 0.1, "TOTAL: 33 DIMENSIONS", fontsize=12,
            fontweight="bold", color=CYAN, ha="center", va="center",
            fontfamily="monospace")

# ── Save ──
out = "/home/user/UniOS/olympex-nautilus/results/state_vector_33dim.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved → {out}")
