from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd


def trim_leading_trailing_zeros(values: np.ndarray) -> np.ndarray:

	if values.size == 0:
		return values

	# Identify non-zero positions
	nonzero_mask = values != 0
	if not nonzero_mask.any():
		# All zeros -> return empty array
		return values[:0]

	first_idx = int(np.argmax(nonzero_mask))
	last_idx = int(values.size - 1 - np.argmax(nonzero_mask[::-1]))
	return values[first_idx : last_idx + 1]


def load_qps_series_from_csv(csv_path: Path) -> np.ndarray:

	# Expect first column is time, remaining columns are per-instance QPS values
	df = pd.read_csv(csv_path)
	if df.shape[1] < 2:
		return np.array([], dtype=float)

	# Drop the first column (time); coerce numeric
	qps_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
	# If there is only one data column, use it directly; otherwise sum them
	if qps_df.shape[1] <= 1:
		values = qps_df.iloc[:, 0].fillna(0.0).to_numpy(dtype=float)
	else:
		row_sum = qps_df.fillna(0.0).sum(axis=1)
		values = row_sum.to_numpy(dtype=float)

	# Trim leading and trailing consecutive zeros
	values = trim_leading_trailing_zeros(values)
	return values


def collect_instance_series(root_dir: Path) -> List[Tuple[str, str, str, np.ndarray]]:

	instances: List[Tuple[str, str, str, np.ndarray]] = []
	for dataset_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
		dataset_name = dataset_dir.name
		for method_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
			method_name = method_dir.name
			csv_files = sorted(method_dir.glob("*.csv"))
			if not csv_files:
				continue
			for csv_path in csv_files:
				# Derive instance id from the header second column if present; fallback to file stem
				try:
					df_head = pd.read_csv(csv_path, nrows=0)
					if df_head.shape[1] >= 2:
						instance_id = str(df_head.columns[1])
					else:
						instance_id = csv_path.stem
				except Exception:
					instance_id = csv_path.stem

				values = load_qps_series_from_csv(csv_path)
				if values.size == 0:
					# Skip files that are entirely zero or invalid after trimming
					continue
				instances.append((dataset_name, method_name, instance_id, values))
	return instances


def compute_stats_long(instances: Iterable[Tuple[str, str, str, np.ndarray]]) -> Tuple[pd.DataFrame, pd.DataFrame]:

	rows_mean: List[Dict[str, object]] = []
	rows_var: List[Dict[str, object]] = []
	for dataset_name, method_name, instance_id, values in instances:
		if values.size == 0:
			continue
		mean_value = float(np.mean(values))
		var_value = float(np.var(values, ddof=0))
		rows_mean.append({"dataset": dataset_name, "method": method_name, "instance": instance_id, "mean": mean_value})
		rows_var.append({"dataset": dataset_name, "method": method_name, "instance": instance_id, "variance": var_value})

	mean_df = pd.DataFrame(rows_mean, columns=["dataset", "method", "instance", "mean"]).sort_values(["dataset", "method", "instance"]).reset_index(drop=True)
	var_df = pd.DataFrame(rows_var, columns=["dataset", "method", "instance", "variance"]).sort_values(["dataset", "method", "instance"]).reset_index(drop=True)
	return mean_df, var_df


def main() -> None:

	parser = argparse.ArgumentParser(description="Compute per-dataset per-method QPS mean and variance (per second).")
	parser.add_argument(
		"--root",
		type=Path,
		default=Path(__file__).resolve().parent / "hete_sched_data",
		help="Root directory containing dataset/method CSV folders (default: ./hete_sched_data)",
	)
	parser.add_argument(
		"--out-prefix",
		type=Path,
		default=None,
		help="Optional output directory; defaults to the root directory.",
	)
	args = parser.parse_args()

	root_dir: Path = args.root
	if not root_dir.exists() or not root_dir.is_dir():
		raise SystemExit(f"Root directory not found or not a directory: {root_dir}")

	instances = collect_instance_series(root_dir)
	if not instances:
		print("No valid QPS data found.")
		mean_df = pd.DataFrame(columns=["dataset", "method", "instance", "mean"])  # empty long tables
		var_df = pd.DataFrame(columns=["dataset", "method", "instance", "variance"])  # empty long tables
	else:
		mean_df, var_df = compute_stats_long(instances)

	output_dir = args.out_prefix if args.out_prefix is not None else root_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	mean_path = output_dir / "qps_mean.csv"
	var_path = output_dir / "qps_variance.csv"
	mean_df.to_csv(mean_path)
	var_df.to_csv(var_path)

	print(f"Wrote mean table: {mean_path}")
	print(f"Wrote variance table: {var_path}")


if __name__ == "__main__":
	main()


