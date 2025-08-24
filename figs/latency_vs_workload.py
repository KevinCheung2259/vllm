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
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.2
})

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

# -------- Plot: two stacked subplots --------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)

# Router's View (Aggregated Latency): dashed black, no markers
ax1.plot(t, latency, linestyle="--", color="red", label="5-sec Avg. Latency")
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Router's View")
ax1.legend(loc="upper right", frameon=False)
# 固定轴范围以匹配目标区间（可选，保证视觉一致）
ax1.set_ylim(20, 70)

# 添加两条纵向虚线
ax1.axvline(x=7.5, color='black', linestyle=':', alpha=0.7, linewidth=1.0)
ax1.axvline(x=11.0, color='black', linestyle=':', alpha=0.7, linewidth=1.0)

# 添加决策延迟标注（水平箭头线）
ax1.annotate("Decision Lag (≈3.5s)",
             xy=(11.0, np.percentile(latency, 80)), xycoords='data',
             xytext=(4.5, np.percentile(latency, 80)), textcoords='data',
             arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2),
             fontsize=9, ha='center', va='center')

# Engine's View (Instantaneous Workload): solid black, no markers
ax2.plot(t, workload, linestyle="-", color="blue", label="Instantaneous Workload (tokens)")
ax2.set_ylabel("Workload (tokens)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Engine's View")
ax2.legend(loc="upper right", frameon=False)
ax2.set_ylim(200, 2000)

# -------- Optional: Decision lag annotation（不遮挡曲线） --------
# 移除原来的决策延迟标注，因为已经在上面添加了
# ax1.annotate("Decision Lag (≈3s)",
#              xy=(10.8, np.percentile(latency, 60)), xycoords='data',
#              xytext=(7.8, np.percentile(latency, 85)), textcoords='data',
#              arrowprops=dict(arrowstyle="<->", color="black"),
#              fontsize=9)

ax2.annotate("Workload Surge (≈7.4s)",
             xy=(7.5, 1900), xycoords='data',
             xytext=(0, 1400), textcoords='data',
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# -------- Save --------
out_path = "vllm/figs/latency_vs_workload.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
