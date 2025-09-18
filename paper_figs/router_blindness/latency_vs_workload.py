#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure X: Latency (Router's View, aggregated) vs. Workload (Engine's View, instantaneous)
Paper-style: black & white, no markers, dashed vs solid, labels placed off curves.
Now with flat-top peaks and value rescaling:
  latency mapped to [20, 70], workload mapped to [200, 2000],
  while preserving the original shapes.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------- Paper-ish style settings --------
def apply_topconf_style():
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
        'axes.linewidth': 1.2,
        'axes.titlesize': 16,
        'axes.labelsize': 18,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'figure.dpi': 160,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        # 轻网格、路径优化
        'grid.alpha': 0.25,
        'path.simplify': True,
        'path.simplify_threshold': 0.5,
        'agg.path.chunksize': 10000,
    })

apply_topconf_style()

# -------- Time axis --------
t = np.linspace(0, 20, 200)
rng = np.random.default_rng(42)

# -------- Helper: smooth flat-top pulse --------
def flat_top_pulse(t, start, end, rise, fall=None):
    """
    Smooth rectangular-like pulse using sigmoids.
    start: plateau start time
    end:   plateau end time
    rise:  rise edge time-scale
    fall:  fall edge time-scale (defaults to rise if None)
    Returns values ~0 outside [start, end], ~1 on the plateau, smooth edges.
    """
    if fall is None:
        fall = rise
    sig_rise = 1.0 / (1.0 + np.exp(-(t - start) / rise))
    sig_fall = 1.0 / (1.0 + np.exp(-(t - end)   / fall))
    return sig_rise * (1.0 - sig_fall)

# -------- Engine's View: instantaneous workload (flat-top surge around ~9s) --------
workload_base = 500.0
workload_amp  = 120.0
workload_env  = flat_top_pulse(t, start=6.6, end=8.4, rise=0.15, fall=0.15)
workload_noise = 15.0 * rng.normal(size=len(t))
workload_raw = workload_base + workload_amp * workload_env + workload_noise

# -------- Router's View: aggregated latency (flat-top, lagging and smoother) --------
lat_base = 30.0
lat_amp  = 40.0
lat_env  = flat_top_pulse(t, start=8.8, end=13.5, rise=0.8, fall=0.8)
lat_noise = 1.5 * rng.normal(size=len(t))
latency_raw = lat_base + lat_amp * lat_env + lat_noise

# -------- Linear rescaling to target ranges while preserving shape --------
def rescale_to_range(x, lo, hi):
    xmin, xmax = np.min(x), np.max(x)
    # 防止极端情况下 xmax == xmin
    if np.isclose(xmax, xmin):
        return np.full_like(x, (lo + hi) / 2.0)
    xnorm = (x - xmin) / (xmax - xmin)
    return lo + xnorm * (hi - lo)

# Map latency to [20, 70], workload to [200, 2000]
latency  = rescale_to_range(latency_raw,  20.0,   70.0)
workload = rescale_to_range(workload_raw, 200.0, 2000.0)

# -------- Plot: single plot with dual Y-axis --------
fig, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)

# 左轴：Engine's View (Instantaneous Workload)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Workload (tokens)", color="blue")
ax1.plot(t, workload, linestyle="-", color="blue", linewidth=2.0, label="Instantaneous Workload (Engine's View)")
ax1.set_ylim(200, 2000)
ax1.tick_params(axis='y', labelcolor='blue')

# 右轴：Router's View (Aggregated Latency)
ax2 = ax1.twinx()
ax2.set_ylabel("Latency (ms)", color="red")
ax2.plot(t, latency, linestyle="--", color="red", linewidth=2.0, label="5-sec Avg. Latency (Router's View)")
ax2.set_ylim(20, 70)
ax2.tick_params(axis='y', labelcolor='red')

# 由于全局关闭了右侧脊线，这里对 twinx 的右脊线单独开启，保持可读
if 'right' in ax2.spines:
    ax2.spines['right'].set_visible(True)

# 添加两条纵向虚线
ax1.axvline(x=7.5, color='black', linestyle=':', alpha=0.7, linewidth=1.0)
ax1.axvline(x=11.0, color='black', linestyle=':', alpha=0.7, linewidth=1.0)

# 添加决策延迟标注（水平箭头线）
ax2.annotate("Decision Lag (≈3.5s)",
             xy=(11.0, np.percentile(latency, 77)), xycoords='data',
             xytext=(5.1, np.percentile(latency, 77)), textcoords='data',
             arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2),
             fontsize=12, ha='center', va='center')

# 工作负载激增标注
ax1.annotate("Workload Surge (≈7.5s)",
             xy=(7.5, 1900), xycoords='data',
             xytext=(1, 1700), textcoords='data',
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=12)

# 添加轻量网格
ax1.grid(True, axis='both', linestyle='-', linewidth=0.4, alpha=0.25)

# 创建组合图例
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color="blue", linestyle="-", linewidth=2.0, label="Instantaneous Workload"),
    Line2D([0], [0], color="red", linestyle="--", linewidth=2.0, label="5-sec Avg. Latency")
]
ax1.legend(handles=custom_lines, loc='upper right', fontsize=13, frameon=False)

# -------- Save --------
out_path = "./latency_vs_workload.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
