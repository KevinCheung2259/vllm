#!/usr/bin/env python3

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AnalysisConfig:
    input_path: str
    output_dir: str
    min_total_scheduled_tokens: int = 2048
    duration_min_ms: float = 42.0
    duration_max_ms: float = 70.0
    top_k_groups: int = 5
    iqr_outlier_threshold: float = 1.5
    min_group_size: int = 2
    bin_width: int = 256
    analyze_bins: bool = True
    compute_partial_spearman: bool = False


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                # 跳过坏行
                continue
    return records


def safe_list_stats(values: Optional[Iterable[float]]) -> Dict[str, float]:
    if values is None:
        values = []
    vals = list(values)
    if len(vals) == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    arr = np.array(vals, dtype=float)
    return {
        "count": float(len(arr)),
        "sum": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "std": float(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def compute_derived_features(record: Dict[str, Any]) -> Dict[str, Any]:
    chunk_sizes = record.get("chunk_sizes") or []
    all_computed_tokens = record.get("all_computed_tokens") or []
    all_cached_tokens = record.get("all_cached_tokens") or []

    chunk_stats = safe_list_stats(chunk_sizes)
    computed_stats = safe_list_stats(all_computed_tokens)
    cached_stats = safe_list_stats(all_cached_tokens)

    computed_sum = computed_stats["sum"]
    cached_sum = cached_stats["sum"]
    total_token_activity = computed_sum + cached_sum
    cache_ratio = (cached_sum / total_token_activity) if total_token_activity > 0 else 0.0

    num_reqs = int(chunk_stats["count"]) if not math.isnan(chunk_stats["count"]) else 0

    features = {
        # 原字段
        "batch_id": record.get("batch_id"),
        "timestamp": record.get("timestamp"),
        "total_scheduled_tokens": record.get("total_scheduled_tokens", 0),
        "schedule_duration_ms": record.get("schedule_duration_ms", 0.0),
        "num_waiting_reqs": record.get("num_waiting_reqs", 0),
        "num_running_reqs": record.get("num_running_reqs", 0),
        "model_run_duration_ms": record.get("model_run_duration_ms", 0.0),
        # 列表统计特征
        "chunk_count": num_reqs,
        "chunk_sum": chunk_stats["sum"],
        "chunk_mean": chunk_stats["mean"],
        "chunk_std": chunk_stats["std"],
        "chunk_p90": chunk_stats["p90"],
        "chunk_p95": chunk_stats["p95"],
        "chunk_max": chunk_stats["max"],
        "computed_sum": computed_sum,
        "computed_mean": computed_stats["mean"],
        "computed_std": computed_stats["std"],
        "computed_p90": computed_stats["p90"],
        "computed_p95": computed_stats["p95"],
        "computed_max": computed_stats["max"],
        "cached_sum": cached_sum,
        "cached_mean": cached_stats["mean"],
        "cached_std": cached_stats["std"],
        "cached_p90": cached_stats["p90"],
        "cached_p95": cached_stats["p95"],
        "cached_max": cached_stats["max"],
        # 派生比率/密度
        "cache_ratio": cache_ratio,
        "avg_computed_per_req": (computed_sum / num_reqs) if num_reqs > 0 else 0.0,
        "avg_cached_per_req": (cached_sum / num_reqs) if num_reqs > 0 else 0.0,
        "avg_chunk_size": (chunk_stats["mean"] if num_reqs > 0 else 0.0),
        # 交互项（粗略反映负载与并行度）
        "computed_sum_x_reqs": computed_sum * max(1, num_reqs),
        "cached_sum_x_reqs": cached_sum * max(1, num_reqs),
    }

    return features


def robust_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    try:
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    except Exception:
        return float("nan")


def group_variance_summary(df: pd.DataFrame, group_col: str = "total_scheduled_tokens") -> pd.DataFrame:
    grouped = df.groupby(group_col)["model_run_duration_ms"]
    summary = grouped.agg([
        ("count", "count"),
        ("min", "min"),
        ("p25", lambda s: np.percentile(s, 25)),
        ("p50", lambda s: np.percentile(s, 50)),
        ("p75", lambda s: np.percentile(s, 75)),
        ("max", "max"),
        ("mean", "mean"),
        ("std", "std"),
        ("iqr", lambda s: np.percentile(s, 75) - np.percentile(s, 25)),
        ("range", lambda s: np.max(s) - np.min(s)),
    ]).reset_index()
    summary["variance_score"] = (
        summary["iqr"].fillna(0.0) * 1.0
        + summary["range"].fillna(0.0) * 0.5
        + summary["std"].fillna(0.0) * 0.25
    )
    summary = summary.sort_values(["variance_score", "count"], ascending=[False, False])
    return summary


def analyze_within_group(df: pd.DataFrame, group_key: Any, top_features: int = 8) -> Dict[str, Any]:
    group_df = df[df["total_scheduled_tokens"] == group_key]
    y = group_df["model_run_duration_ms"].to_numpy()
    n = len(group_df)
    if n == 0:
        return {"group": group_key, "count": 0, "message": "该组无数据"}

    candidate_features = [
        "num_running_reqs",
        "num_waiting_reqs",
        "schedule_duration_ms",
        "chunk_count",
        "chunk_sum",
        "chunk_mean",
        "chunk_std",
        "chunk_p95",
        "chunk_max",
        "computed_sum",
        "computed_mean",
        "computed_std",
        "computed_p95",
        "computed_max",
        "cached_sum",
        "cached_mean",
        "cached_std",
        "cached_p95",
        "cached_max",
        "cache_ratio",
        "avg_computed_per_req",
        "avg_cached_per_req",
        "avg_chunk_size",
        "computed_sum_x_reqs",
        "cached_sum_x_reqs",
    ]

    correlations: List[Tuple[str, float]] = []
    if n >= 3:
        for feat in candidate_features:
            if feat not in group_df.columns:
                continue
            x = group_df[feat].to_numpy(dtype=float)
            r = robust_spearmanr(x, y)
            correlations.append((feat, r))
        correlations = [(k, v) for k, v in correlations if not (isinstance(v, float) and math.isnan(v))]
        correlations_sorted = sorted(correlations, key=lambda kv: (abs(kv[1]), kv[1]), reverse=True)[:top_features]
    else:
        correlations_sorted = []

    # 分位数切片（n 太小时退化处理）
    if n >= 4:
        high_mask = y >= np.percentile(y, 75)
        low_mask = y <= np.percentile(y, 25)
        high_df = group_df[high_mask]
        low_df = group_df[low_mask]
    else:
        # 取 top/bottom by y 的各一半
        order = np.argsort(y)
        half = max(1, n // 2)
        low_idx = order[:half]
        high_idx = order[-half:]
        low_df = group_df.iloc[low_idx]
        high_df = group_df.iloc[high_idx]

    def feature_diff(feat: str) -> Tuple[float, float, float]:
        if feat not in group_df.columns:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(np.mean(high_df[feat])) if len(high_df) > 0 else float("nan"),
            float(np.mean(low_df[feat])) if len(low_df) > 0 else float("nan"),
            float((np.mean(high_df[feat]) - np.mean(low_df[feat])) if len(high_df) > 0 and len(low_df) > 0 else float("nan")),
        )

    diffs = []
    for feat, r in correlations_sorted if correlations_sorted else [(f, float("nan")) for f in candidate_features[:8]]:
        high_mean, low_mean, delta = feature_diff(feat)
        diffs.append({
            "feature": feat,
            "spearman_r": float(r) if not (isinstance(r, float) and math.isnan(r)) else None,
            "high_p75_mean": high_mean,
            "low_p25_mean": low_mean,
            "delta": delta,
        })

    duration_stats = {
        "min": float(np.min(y)),
        "p25": float(np.percentile(y, 25)) if n >= 4 else float(np.min(y)),
        "p50": float(np.percentile(y, 50)),
        "p75": float(np.percentile(y, 75)) if n >= 4 else float(np.max(y)),
        "max": float(np.max(y)),
        "std": float(np.std(y, ddof=1)) if n >= 2 else 0.0,
        "iqr": float(np.percentile(y, 75) - np.percentile(y, 25)) if n >= 4 else float(np.max(y) - np.min(y)),
    }

    return {
        "group": group_key,
        "count": int(n),
        "duration_stats": duration_stats,
        "top_feature_correlations": [
            {"feature": feat, "spearman_r": (float(r) if r is not None and not math.isnan(r) else None)} for feat, r in correlations_sorted
        ],
        "feature_diffs_high_vs_low": diffs,
    }


def global_feature_analysis(df: pd.DataFrame, top_features: int = 12) -> Dict[str, Any]:
    y = df["model_run_duration_ms"].to_numpy()
    candidate_features = [
        "total_scheduled_tokens",
        "num_running_reqs",
        "num_waiting_reqs",
        "schedule_duration_ms",
        "chunk_count",
        "chunk_sum",
        "chunk_mean",
        "chunk_std",
        "chunk_p95",
        "chunk_max",
        "computed_sum",
        "computed_mean",
        "computed_std",
        "computed_p90",
        "computed_p95",
        "computed_max",
        "cached_sum",
        "cached_mean",
        "cached_std",
        "cached_p90",
        "cached_p95",
        "cached_max",
        "cache_ratio",
        "avg_computed_per_req",
        "avg_cached_per_req",
        "avg_chunk_size",
        "computed_sum_x_reqs",
        "cached_sum_x_reqs",
    ]
    corrs = []
    for feat in candidate_features:
        if feat not in df.columns:
            continue
        x = df[feat].to_numpy(dtype=float)
        r = robust_spearmanr(x, y)
        if not (isinstance(r, float) and math.isnan(r)):
            corrs.append((feat, float(r)))
    corrs_sorted = sorted(corrs, key=lambda kv: (abs(kv[1]), kv[1]), reverse=True)[:top_features]

    # 简单多元线性回归（标准化）
    feats = [f for f, _ in corrs_sorted]
    X = df[feats].to_numpy(dtype=float)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xn = (X - X_mean) / X_std
    Yn = y
    try:
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xn)), Xn], Yn, rcond=None)
        intercept = float(coef[0])
        betas = coef[1:]
        # 计算 R^2
        y_hat = np.c_[np.ones(len(Xn)), Xn] @ coef
        ss_res = float(np.sum((Yn - y_hat) ** 2))
        ss_tot = float(np.sum((Yn - np.mean(Yn)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    except Exception:
        intercept = 0.0
        betas = np.zeros(len(feats))
        r2 = float("nan")

    coefs = [{"feature": f, "beta": float(b)} for f, b in zip(feats, betas)]
    coefs_sorted = sorted(coefs, key=lambda d: abs(d["beta"]), reverse=True)

    return {
        "spearman_top": [{"feature": f, "spearman_r": r} for f, r in corrs_sorted],
        "linear_regression": {
            "features": feats,
            "intercept": intercept,
            "betas": coefs_sorted,
            "r2": float(r2),
        },
    }


def _linear_residual(z: np.ndarray, control: np.ndarray) -> np.ndarray:
    n = len(z)
    A = np.c_[np.ones(n), control]
    try:
        beta, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ beta
    except Exception:
        resid = z - np.mean(z)
    return resid


def compute_partial_spearman(
    df: pd.DataFrame,
    y_col: str,
    control_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    # 排名变换以逼近 Spearman
    tmp = df[[y_col, control_col] + [c for c in feature_cols if c in df.columns]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(tmp) < 3:
        return pd.DataFrame(columns=["feature", "partial_spearman_r", "n"]).set_index("feature")

    y_rank = tmp[y_col].rank(method="average").to_numpy(dtype=float)
    t_rank = tmp[control_col].rank(method="average").to_numpy(dtype=float)
    y_resid = _linear_residual(y_rank, t_rank)

    results: List[Tuple[str, float, int]] = []
    for feat in feature_cols:
        if feat not in tmp.columns:
            continue
        x_rank = tmp[feat].rank(method="average").to_numpy(dtype=float)
        x_resid = _linear_residual(x_rank, t_rank)
        if np.std(x_resid) == 0 or np.std(y_resid) == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(x_resid, y_resid)[0, 1])
        results.append((feat, r, len(tmp)))

    out = pd.DataFrame(results, columns=["feature", "partial_spearman_r", "n"]).set_index("feature")
    out = out.sort_values(by="partial_spearman_r", key=lambda s: s.abs(), ascending=False)
    return out


def try_plot(df: pd.DataFrame, output_dir: str, feature_pairs: List[Tuple[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for x_col, y_col in feature_pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x_col], df[y_col], s=10, alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.grid(True, alpha=0.3)
        fig_path = os.path.join(output_dir, f"scatter_{y_col}_vs_{x_col}.png")
        try:
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
        except Exception:
            pass
        finally:
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scheduler profiling JSONL")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result/scheduler_profiling_chunk_4096.jsonl",
        help="输入 JSONL 路径",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/paperspace/zhangy/vllm-workspace/vllm/predict_exp/outputs",
        help="输出目录",
    )
    parser.add_argument("--min_tokens", type=int, default=2048, help="total_scheduled_tokens 下限(含)")
    parser.add_argument("--dur_min", type=float, default=42.0, help="model_run_duration_ms 下界(含)")
    parser.add_argument("--dur_max", type=float, default=70.0, help="model_run_duration_ms 上界(含)")
    parser.add_argument("--topk", type=int, default=5, help="分析方差最大的前 K 个 token 组")
    parser.add_argument("--min_group_size", type=int, default=2, help="组内最小样本数，用于不再轻易跳过分析")
    parser.add_argument("--bin_width", type=int, default=256, help="按 total_scheduled_tokens 分箱的宽度")
    parser.add_argument("--no_bins", action="store_true", help="关闭分箱分析")
    parser.add_argument("--partial_spearman", action="store_true", help="计算控制 total_scheduled_tokens 的偏相关(Spearman)")
    args = parser.parse_args()

    cfg = AnalysisConfig(
        input_path=args.input,
        output_dir=args.outdir,
        min_total_scheduled_tokens=args.min_tokens,
        duration_min_ms=args.dur_min,
        duration_max_ms=args.dur_max,
        top_k_groups=args.topk,
        min_group_size=args.min_group_size,
        bin_width=args.bin_width,
        analyze_bins=(not args.no_bins),
        compute_partial_spearman=args.partial_spearman,
    )

    ensure_dir(cfg.output_dir)

    # 读取与扁平化
    raw_records = read_jsonl(cfg.input_path)
    flat_records = [compute_derived_features(r) for r in raw_records]
    df = pd.DataFrame(flat_records)

    # 基本筛选
    mask = (
        (df["total_scheduled_tokens"] > cfg.min_total_scheduled_tokens)
        & (df["model_run_duration_ms"] >= cfg.duration_min_ms)
        & (df["model_run_duration_ms"] <= cfg.duration_max_ms)
    )
    filtered_df = df[mask].copy()

    # 输出筛选结果
    filtered_path = os.path.join(cfg.output_dir, "filtered_records.csv")
    filtered_df.to_csv(filtered_path, index=False)

    # 组内差异汇总
    variance_df = group_variance_summary(filtered_df, group_col="total_scheduled_tokens")
    variance_path = os.path.join(cfg.output_dir, "per_total_tokens_variance.csv")
    variance_df.to_csv(variance_path, index=False)

    # 选择方差大的若干组深入分析（即使样本很少也进行对比）
    head_groups = variance_df.head(cfg.top_k_groups)["total_scheduled_tokens"].tolist()
    analyses: List[Dict[str, Any]] = []
    for g in head_groups:
        analyses.append(analyze_within_group(filtered_df, g, top_features=10))

    # 分箱分析
    bin_analyses: List[Dict[str, Any]] = []
    variance_bin_df: Optional[pd.DataFrame] = None
    if cfg.analyze_bins:
        filtered_df["total_tokens_bin"] = (
            (filtered_df["total_scheduled_tokens"] // cfg.bin_width) * cfg.bin_width
        )
        variance_bin_df = group_variance_summary(filtered_df, group_col="total_tokens_bin")
        variance_bin_path = os.path.join(cfg.output_dir, f"per_token_bins_variance_w{cfg.bin_width}.csv")
        variance_bin_df.to_csv(variance_bin_path, index=False)
        head_bins = variance_bin_df.head(cfg.top_k_groups)["total_tokens_bin"].tolist()
        for b in head_bins:
            # 将组键改写为该箱的所有样本的总 token 值范围提示
            bin_df = filtered_df[filtered_df["total_tokens_bin"] == b].copy()
            # 临时将该箱样本的 total_scheduled_tokens 统一写入 b，以复用 analyze_within_group
            original = bin_df["total_scheduled_tokens"].copy()
            bin_df["total_scheduled_tokens"] = b
            # 调用分析
            res = analyze_within_group(bin_df, b, top_features=10)
            # 恢复（仅报告用途）
            res["bin_width"] = cfg.bin_width
            res["bin_value"] = int(b)
            res["bin_token_range_min"] = int(original.min()) if len(original) else int(b)
            res["bin_token_range_max"] = int(original.max()) if len(original) else int(b)
            bin_analyses.append(res)

    # 全局分析（不分组）
    global_analysis = global_feature_analysis(filtered_df, top_features=12)

    # 导出 JSON 报告
    report_json_path = os.path.join(cfg.output_dir, "group_analyses.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "by_exact_total_tokens": analyses,
            "by_token_bins": bin_analyses,
            "global": global_analysis,
        }, f, ensure_ascii=False, indent=2)

    # 导出可读 Markdown 报告
    md_lines: List[str] = []
    md_lines.append(f"输入: `{cfg.input_path}`\n")
    md_lines.append(f"筛选条件: total_scheduled_tokens > {cfg.min_total_scheduled_tokens}, model_run_duration_ms ∈ [{cfg.duration_min_ms}, {cfg.duration_max_ms}]\n")
    md_lines.append(f"筛选后样本数: {len(filtered_df)}\n")

    md_lines.append("\n## 全局特征影响 (Spearman 与线性回归)\n")
    md_lines.append("- Spearman Top 特征:\n")
    for item in global_analysis["spearman_top"]:
        md_lines.append(f"  - {item['feature']}: r={item['spearman_r']:.3f}\n")
    md_lines.append(f"- 线性回归: R^2={global_analysis['linear_regression']['r2']:.3f}, 重要系数(绝对值排序):\n")
    for item in global_analysis["linear_regression"]["betas"][:10]:
        md_lines.append(f"  - {item['feature']}: beta={item['beta']:.4f}\n")

    md_lines.append("\n## 按精确 total_scheduled_tokens 分组\n")
    for a in analyses:
        md_lines.append(f"\n---\n")
        md_lines.append(f"### total_scheduled_tokens = {a.get('group')} (n={a.get('count')})\n")
        if "message" in a and a.get("count", 0) == 0:
            md_lines.append(f"{a['message']}\n")
            continue
        ds = a["duration_stats"]
        md_lines.append(
            f"耗时分布: min={ds['min']:.3f}, p25={ds['p25']:.3f}, p50={ds['p50']:.3f}, p75={ds['p75']:.3f}, max={ds['max']:.3f}, std={ds['std']:.3f}, iqr={ds['iqr']:.3f}\n"
        )
        if a.get("top_feature_correlations"):
            md_lines.append("可能的关键因素(按|spearman_r|排序):\n")
            for item in a["top_feature_correlations"]:
                r = item['spearman_r']
                r_str = (f"{r:.3f}" if isinstance(r, float) else "NA")
                md_lines.append(f"- {item['feature']}: spearman_r={r_str}\n")
        md_lines.append("与高耗时组~低耗时组的均值差异(退化为上下半区时也类似):\n")
        for item in a["feature_diffs_high_vs_low"]:
            md_lines.append(
                f"- {item['feature']}: high_mean={item['high_p75_mean']:.3f}, low_mean={item['low_p25_mean']:.3f}, delta={item['delta']:.3f}\n"
            )

    if cfg.analyze_bins and variance_bin_df is not None and len(variance_bin_df) > 0:
        md_lines.append("\n## 按 total_scheduled_tokens 分箱\n")
        md_lines.append(f"分箱宽度: {cfg.bin_width}, 统计文件: per_token_bins_variance_w{cfg.bin_width}.csv\n")
        for a in bin_analyses:
            md_lines.append(f"\n---\n")
            md_lines.append(
                f"### tokens_bin = {a.get('bin_value')} (range≈[{a.get('bin_token_range_min')}, {a.get('bin_token_range_max')}], n={a.get('count')})\n"
            )
            ds = a.get("duration_stats", {})
            if ds:
                md_lines.append(
                    f"耗时分布: min={ds['min']:.3f}, p25={ds['p25']:.3f}, p50={ds['p50']:.3f}, p75={ds['p75']:.3f}, max={ds['max']:.3f}, std={ds['std']:.3f}, iqr={ds['iqr']:.3f}\n"
                )
            if a.get("top_feature_correlations"):
                md_lines.append("可能的关键因素(按|spearman_r|排序):\n")
                for item in a["top_feature_correlations"]:
                    r = item['spearman_r']
                    r_str = (f"{r:.3f}" if isinstance(r, float) else "NA")
                    md_lines.append(f"- {item['feature']}: spearman_r={r_str}\n")
            md_lines.append("与高耗时组~低耗时组的均值差异:\n")
            for item in a.get("feature_diffs_high_vs_low", [])[:10]:
                md_lines.append(
                    f"- {item['feature']}: high_mean={item['high_p75_mean']:.3f}, low_mean={item['low_p25_mean']:.3f}, delta={item['delta']:.3f}\n"
                )

    report_md_path = os.path.join(cfg.output_dir, "group_analyses.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    # 可选绘图
    try_plot(
        filtered_df,
        cfg.output_dir,
        feature_pairs=[
            ("computed_sum", "model_run_duration_ms"),
            ("cached_sum", "model_run_duration_ms"),
            ("cache_ratio", "model_run_duration_ms"),
            ("num_running_reqs", "model_run_duration_ms"),
            ("chunk_count", "model_run_duration_ms"),
            ("avg_computed_per_req", "model_run_duration_ms"),
        ],
    )

    # 偏相关（控制 total_scheduled_tokens）
    if cfg.compute_partial_spearman:
        factors = [
            "num_running_reqs", "num_waiting_reqs", "schedule_duration_ms",
            "chunk_count", "chunk_sum", "chunk_mean", "chunk_std", "chunk_p90", "chunk_p95", "chunk_max",
            "computed_sum", "computed_mean", "computed_std", "computed_p90", "computed_p95", "computed_max",
            "cached_sum", "cached_mean", "cached_std", "cached_p90", "cached_p95", "cached_max",
            "cache_ratio", "avg_computed_per_req", "avg_cached_per_req", "avg_chunk_size",
            "computed_sum_x_reqs", "cached_sum_x_reqs",
        ]
        partial_df = compute_partial_spearman(
            filtered_df,
            y_col="model_run_duration_ms",
            control_col="total_scheduled_tokens",
            feature_cols=factors,
        )
        partial_csv = os.path.join(cfg.output_dir, "partial_spearman_control_total_tokens.csv")
        partial_df.to_csv(partial_csv)
        print("偏相关(控制 total_scheduled_tokens)保存:", partial_csv)
        # 控制台打印前 15 项
        try:
            print(partial_df.head(15).to_string())
        except Exception:
            pass

    print("筛选结果保存:", filtered_path)
    print("组内方差汇总保存:", variance_path)
    if cfg.analyze_bins and variance_bin_df is not None:
        print("分箱方差汇总保存:", variance_bin_path)
    print("组内/分箱/全局分析(JSON):", report_json_path)
    print("可读报告(MD):", report_md_path)


if __name__ == "__main__":
    main() 