#!/usr/bin/env python3
"""Compare R² over time: partial_fit (online adaptation) vs static (no adaptation).

Demonstrates that the pretrained model alone cannot track workload changes,
while partial_fit maintains high prediction accuracy across QPS regime shifts.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
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

SKIP_TRANSIENT = 3


def load_r2_trace(path: Path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_r2_comparison(partial_csv: Path, static_csv: Path, qps_phases=None):
    partial_rows = load_r2_trace(partial_csv)[SKIP_TRANSIENT:]
    static_rows = load_r2_trace(static_csv)[SKIP_TRANSIENT:]

    n_partial = len(partial_rows)
    n_static = len(static_rows)
    n = min(n_partial, n_static)

    updates = list(range(1, n + 1))
    r2_partial = [float(r["r2"]) for r in partial_rows[:n]]
    r2_static = [float(r["r2"]) for r in static_rows[:n]]

    # --- Figure: R² comparison ---
    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)

    ax.plot(updates, r2_partial, color="#2196F3", marker="o", markersize=4,
            linewidth=1.5, label="With online adaptation (partial fit)")
    ax.plot(updates, r2_static, color="#E53935", marker="s", markersize=4,
            linewidth=1.5, label="Static pretrained model (no adaptation)")

    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel(r"$R^2$")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Add QPS phase annotations
    if qps_phases:
        ylims = ax.get_ylim()
        for phase_idx, phase_label in qps_phases:
            if 1 <= phase_idx <= n:
                ax.axvline(phase_idx, color="red", linewidth=1.0,
                           linestyle="--", alpha=0.5)
                ax.text(phase_idx + 0.3, ylims[1] * 0.95, phase_label,
                        fontsize=8, color="red", alpha=0.8, va="top")

    out_path = HERE / "r2_comparison_partial_vs_static.pdf"
    fig.savefig(out_path)
    print(f"Saved R² comparison to: {out_path}")

    # Print summary
    print(f"\n=== R² Summary ===")
    print(f"{'Mode':<30} {'Mean R²':>10} {'Min R²':>10} {'Max R²':>10}")
    print("-" * 62)
    print(f"{'Partial fit (adaptation)':<30} {np.mean(r2_partial):>10.3f} {np.min(r2_partial):>10.3f} {np.max(r2_partial):>10.3f}")
    print(f"{'Static pretrained (no adapt)':<30} {np.mean(r2_static):>10.3f} {np.min(r2_static):>10.3f} {np.max(r2_static):>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare R² between partial_fit and static model")
    parser.add_argument("--partial", type=str, default=str(HERE / "param_trace_partial.csv"))
    parser.add_argument("--static", type=str, default=str(HERE / "param_trace_evalonly.csv"))
    parser.add_argument("--phases", type=str, default=None,
                        help="QPS phase boundaries as 'idx1:label1,idx2:label2,...'")
    args = parser.parse_args()

    qps_phases = None
    if args.phases:
        qps_phases = []
        for item in args.phases.split(","):
            idx_str, label = item.split(":")
            qps_phases.append((int(idx_str), label))

    plot_r2_comparison(Path(args.partial), Path(args.static), qps_phases)
