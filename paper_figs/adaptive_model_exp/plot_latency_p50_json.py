#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study: p50 latency comparison across scheduling strategies.
Style aligned with plot_latency_comparison.py.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    "axes.linewidth": 1.2,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

HERE = Path(__file__).resolve().parent

ROUND_SEC  = 10
TIME_MAX   = 90
MAX_ROUNDS = TIME_MAX // ROUND_SEC   # 9
SPLIT      = 25
MARK_EVERY = 2

JSON_FILES = [
    ("adaptive-qps5.json",      "Online-MT"),
    ("offline-qps5.json",       "Offline"),
    ("native-qps5.json",        "No-model"),
    ("long_window-qps5.json",   "Online-LW"),
    ("short_window-qps5.json",  "Online-SW"),
]

COLORS: Dict[str, str] = {
    "Online-MT":          "orange",
    "Offline":            "green",
    "No-model":             "blue",
    "Online-LW":   "red",
    "Online-SW":  "purple",
}
LINESTYLES: Dict[str, str] = {
    "Online-MT":          "-",
    "Offline":            "--",
    "No-model":             "-.",
    "Online-LW":   ":",
    "Online-SW":  "--",
}
MARKERS: Dict[str, str] = {
    "Online-MT":          "o",
    "Offline":            "s",
    "No-model":             "^",
    "Online-LW":   "D",
    "Online-SW":  "v",
}


def load_p50_series(path: Path) -> List[float]:
    p50s: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p50 = (rec.get("Latency") or {}).get("p50")
            if p50 is not None:
                p50s.append(float(p50))
    return p50s


def plot_latency_p50():
    fig, ax = plt.subplots(figsize=(5.6, 3), constrained_layout=True)

    # background shading
    ax.axvspan(-2, SPLIT, facecolor="#f2dede", alpha=0.35, zorder=0)
    ax.axvspan(SPLIT, TIME_MAX + 2, facecolor="#dff0d8", alpha=0.35, zorder=0)
    ax.axvline(x=SPLIT, color="k", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.text(SPLIT / 2.0, 0.95, "Learning Period",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10, color="#444444", fontweight="bold")
    ax.text((SPLIT + TIME_MAX) / 2.0, 0.95, "Stable Period",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10, color="#444444", fontweight="bold")

    for fname, label in JSON_FILES:
        fp = HERE / fname
        if not fp.exists():
            print(f"[WARN] skip {fp}")
            continue
        series = load_p50_series(fp)[:MAX_ROUNDS]
        if not series:
            continue
        x = [i * ROUND_SEC + ROUND_SEC / 2.0 for i in range(len(series))]
        ax.plot(x, series,
                LINESTYLES.get(label, "-"),
                color=COLORS.get(label, "#444"),
                marker=MARKERS.get(label, "o"),
                markevery=MARK_EVERY,
                markersize=4.9,
                linewidth=2.5,
                label=label,
                alpha=0.95,
                zorder=2)

    ax.set_xlim(-2, 92)
    ax.set_ylim(2.6, 6.2)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("p50 Latency (s)")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)

    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, 1.2),
              ncol=5,
              fontsize=9.5,
              frameon=False,
              handlelength=1.8,
              handletextpad=0.5,
              columnspacing=0.8)

    out_path = HERE / "online_latency_p50_qps5.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_latency_p50()
