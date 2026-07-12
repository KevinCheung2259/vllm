#!/usr/bin/env python3
"""Plot parameter stability analysis from param_trace CSV.

For partial_fit mode: structural parameters (P_max, k_B, k_S) are fixed from
offline profiling, while workload-adaptive parameters (w_1, tau_B, tau_S) are
updated online. This demonstrates the multi-timescale adaptation approach.

Uses Coefficient of Variation (CV) as the stability metric.
Supports dynamic QPS experiments with phase annotations.
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

# Skip first N transient updates (buffer warm-up) for stability analysis
SKIP_TRANSIENT = 3

# Parameter grouping for partial_fit mode:
# Structural params are fixed from offline profiling (hardware properties)
# Workload-adaptive params are updated online
STRUCTURAL = ["P_max", "k_B", "k_S"]   # fixed from offline profiling
LINEAR = ["w_1", "tau_B", "tau_S"]      # adapted online


def load_param_trace(path: Path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_cv(values):
    """Coefficient of Variation = std / |mean|"""
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if abs(mean) < 1e-12:
        return float("inf")
    return np.std(arr) / abs(mean)


def plot_param_stability(csv_file: Path, qps_phases=None):
    """Plot parameter stability analysis.

    Args:
        csv_file: Path to the param_trace CSV file.
        qps_phases: Optional list of (update_index, qps_label) tuples for phase annotations.
                    update_index is 1-based (after skipping transient).
    """
    rows = load_param_trace(csv_file)
    if len(rows) <= SKIP_TRANSIENT:
        print(f"Not enough data: {len(rows)} rows, need > {SKIP_TRANSIENT}")
        return

    # Use data after transient
    stable_rows = rows[SKIP_TRANSIENT:]

    # Compute CV for all tracked parameters
    cv_values = {}
    for p in STRUCTURAL + LINEAR:
        values = [float(r[p]) for r in stable_rows]
        cv_values[p] = compute_cv(values)

    # --- Figure 1: CV bar chart ---
    fig, ax = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)

    struct_cvs = [cv_values[p] for p in STRUCTURAL]
    linear_cvs = [cv_values[p] for p in LINEAR]

    struct_labels = [r"$P_{\max}$", r"$k_B$", r"$k_S$"]
    linear_labels_disp = [r"$w_1$", r"$\tau_B$", r"$\tau_S$"]

    n = max(len(struct_labels), len(linear_labels_disp))
    x = np.arange(n)
    width = 0.35

    ax.bar(x[:len(struct_cvs)] - width / 2, struct_cvs, width,
           color="#2196F3", alpha=0.85, label="Structural (offline)",
           edgecolor="white", linewidth=0.5)
    ax.bar(x[:len(linear_cvs)] + width / 2, linear_cvs, width,
           color="#FF9800", alpha=0.85, label="Workload-Adaptive (online)",
           edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Coefficient of Variation")
    ax.set_xticks(x)
    xtick_labels = []
    for i in range(n):
        s_lbl = struct_labels[i] if i < len(struct_labels) else ""
        l_lbl = linear_labels_disp[i] if i < len(linear_labels_disp) else ""
        if s_lbl and l_lbl:
            xtick_labels.append(f"{s_lbl} / {l_lbl}")
        else:
            xtick_labels.append(s_lbl or l_lbl)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)

    out_path = csv_file.parent / "param_stability_cv_partial.pdf"
    fig.savefig(out_path)
    print(f"Saved CV bar chart to: {out_path}")

    # --- Figure 2: Parameter time series (normalized) ---
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.2),
                                     constrained_layout=True, sharex=True)

    updates = list(range(1, len(stable_rows) + 1))

    # Structural parameters (top panel): P_max, k_B, k_S
    colors_s = ["#1565C0", "#42A5F5", "#81D4FA"]
    markers_s = ["D", "s", "^"]
    struct_plot_labels = [r"$P_{\max}$", r"$k_B$", r"$k_S$"]
    for i, (p, label) in enumerate(zip(STRUCTURAL, struct_plot_labels)):
        vals = np.array([float(r[p]) for r in stable_rows])
        mean_val = np.mean(vals)
        if mean_val != 0:
            norm_vals = (vals - mean_val) / mean_val
        else:
            norm_vals = vals
        ax1.plot(updates, norm_vals, color=colors_s[i], marker=markers_s[i],
                 markersize=3.5, linewidth=1.2, label=label)

    ax1.set_ylabel("Normalized Deviation")
    ax1.set_title("Structural Parameters (fixed from offline profiling)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right", ncol=3)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    # Force same y-scale as bottom panel so flatness is visually clear
    ax1.set_ylim(-0.7, 0.7)

    # Workload-adaptive parameters (bottom panel): w_1, tau_B, tau_S
    colors_l = ["#1B5E20", "#E65100", "#FF9800"]
    markers_l = ["D", "o", "s"]
    linear_plot_labels = [r"$w_1$", r"$\tau_B$", r"$\tau_S$"]
    for i, (p, label) in enumerate(zip(LINEAR, linear_plot_labels)):
        vals = np.array([float(r[p]) for r in stable_rows])
        mean_val = np.mean(vals)
        if mean_val != 0:
            norm_vals = (vals - mean_val) / mean_val
        else:
            norm_vals = vals
        ax2.plot(updates, norm_vals, color=colors_l[i], marker=markers_l[i],
                 markersize=3.5, linewidth=1.2, label=label)

    ax2.set_xlabel("Model Update Index")
    ax2.set_ylabel("Normalized Deviation")
    ax2.set_title("Workload-Adaptive Parameters (online partial fit)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right", ncol=3)
    ax2.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Add QPS phase annotations (vertical dashed lines + text at top)
    if qps_phases:
        for cur_ax in (ax1, ax2):
            ylims = cur_ax.get_ylim()
            for phase_idx, phase_label in qps_phases:
                if 1 <= phase_idx <= len(updates):
                    cur_ax.axvline(phase_idx, color="red", linewidth=1.0,
                                   linestyle="--", alpha=0.6)
                    cur_ax.text(phase_idx + 0.3, ylims[1] * 0.92, phase_label,
                                fontsize=8, color="red", alpha=0.8, va="top")

    out_path2 = csv_file.parent / "param_stability_timeseries_partial.pdf"
    fig2.savefig(out_path2)
    print(f"Saved time series to: {out_path2}")

    # Print CV summary
    print("\n=== Coefficient of Variation Summary ===")
    print(f"{'Parameter':<10} {'CV':>8}  {'Group'}")
    print("-" * 40)
    for p in STRUCTURAL:
        print(f"{p:<10} {cv_values[p]:>8.3f}  Structural (offline)")
    for p in LINEAR:
        print(f"{p:<10} {cv_values[p]:>8.3f}  Workload-Adaptive (online)")

    struct_cvs_final = [cv_values[p] for p in STRUCTURAL]
    linear_cvs_final = [cv_values[p] for p in LINEAR]
    print(f"\nMean CV - Structural:        {np.mean(struct_cvs_final):.3f}")
    print(f"Mean CV - Workload-Adaptive: {np.mean(linear_cvs_final):.3f}")
    if np.mean(struct_cvs_final) > 0:
        print(f"Ratio (Adaptive/Structural): {np.mean(linear_cvs_final)/np.mean(struct_cvs_final):.1f}x")
    else:
        print(f"Ratio: inf (structural params are constant)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot parameter stability analysis")
    parser.add_argument("--csv", type=str, default=str(HERE / "param_trace_partial.csv"),
                        help="Path to param_trace CSV file (default: param_trace_partial.csv)")
    parser.add_argument("--phases", type=str, default=None,
                        help="QPS phase boundaries as 'idx1:label1,idx2:label2,...' "
                             "e.g. '1:QPS=3,13:QPS=5,25:QPS=3'")
    args = parser.parse_args()

    csv_path = Path(args.csv)

    qps_phases = None
    if args.phases:
        qps_phases = []
        for item in args.phases.split(","):
            idx_str, label = item.split(":")
            qps_phases.append((int(idx_str), label))

    plot_param_stability(csv_path, qps_phases=qps_phases)
