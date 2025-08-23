#!/usr/bin/env python3
"""
根据 profiling jsonl 分析并可视化：
当 total_scheduled_tokens == 1 时，model_run_duration_ms 随着 all_computed_tokens 的变化而变化。

使用方式：
  python plot_model_run_vs_computed_tokens.py [PATH ...]

PATH 可以是：
- 指向单个 .jsonl 文件
- 指向包含若干 .jsonl 文件的目录
- 如未提供 PATH：默认收集所有可发现的 .jsonl 文件，来源包括
  - ./（当前工作目录）
  - ./exp/profiling_result/
  - ./profiling_result/

输出图：model_run_vs_all_computed_tokens_tst1.png
"""

import sys
import json
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(jsonl_path: Path) -> pd.DataFrame:
    records: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    if not records:
        raise RuntimeError(f"未能从 {jsonl_path} 解析到有效 JSON 行")
    return pd.DataFrame(records)


def collect_jsonl_paths(argv: List[str]) -> List[Path]:
    # 若提供参数：支持文件或目录，均可传入多个
    if argv:
        jsonl_paths: List[Path] = []
        for arg in argv:
            p = Path(arg).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"路径不存在: {p}")
            if p.is_dir():
                files = sorted(p.glob("*.jsonl"))
                if not files:
                    raise FileNotFoundError(f"目录内未找到任何 .jsonl 文件: {p}")
                jsonl_paths.extend(files)
            elif p.is_file():
                if p.suffix.lower() != ".jsonl":
                    raise ValueError(f"不是 .jsonl 文件: {p}")
                jsonl_paths.append(p)
            else:
                raise ValueError(f"不支持的路径类型: {p}")
        return jsonl_paths

    # 未提供参数：默认收集所有可发现的 .jsonl 文件
    jsonl_paths: List[Path] = []
    cwd = Path.cwd()
    # 当前目录下的所有 .jsonl
    jsonl_paths.extend(sorted(cwd.glob("*.jsonl")))
    # exp/profiling_result/*.jsonl（假定从仓库根目录运行）
    exp_dir = cwd / "exp" / "profiling_result"
    if exp_dir.exists():
        jsonl_paths.extend(sorted(exp_dir.glob("*.jsonl")))
    # profiling_result/*.jsonl（与脚本同级或当前目录下）
    pr_dir = cwd / "profiling_result"
    if pr_dir.exists():
        jsonl_paths.extend(sorted(pr_dir.glob("*.jsonl")))

    # 去重并保持顺序
    if jsonl_paths:
        jsonl_paths = list(dict.fromkeys(jsonl_paths))

    if not jsonl_paths:
        raise FileNotFoundError(
            "未找到任何可用的 profiling .jsonl 文件。请显式传入文件或目录，"
            "或将文件放置在当前目录、./exp/profiling_result/ 或 ./profiling_result/ 下。"
        )
    return jsonl_paths


def main():
    jsonl_paths = collect_jsonl_paths(sys.argv[1:])
    df = pd.concat([load_jsonl(p) for p in jsonl_paths], ignore_index=True)

    # 仅保留包含所需字段的记录
    required_cols = ["total_scheduled_tokens", "model_run_duration_ms", "all_computed_tokens"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"数据缺少必要字段: {col}")

    # 过滤全为decode的批次（可切换为下列任一过滤条件）
    # df = df[df["total_scheduled_tokens"].eq(df["all_computed_tokens"].apply(len))]
    df = df[df["total_scheduled_tokens"] == 1]
    if df.empty:
        raise RuntimeError("过滤后无数据")
    # 过滤离群点（可选）
    df = df[df["model_run_duration_ms"] < 500]

    # 构造散点数据：将 all_computed_tokens（列表）展开，与该批次的 model_run_duration_ms 配对
    xs: List[float] = []  # all_computed_tokens 中的数值
    ys: List[float] = []  # 对应的 model_run_duration_ms
    for _, row in df.iterrows():
        mrd = row.get("model_run_duration_ms")
        computed = row.get("all_computed_tokens", [])
        if isinstance(computed, list):
            for val in computed:
                if isinstance(val, (int, float)) and pd.notna(val) and pd.notna(mrd):
                    xs.append(float(val))
                    ys.append(float(mrd))
        elif isinstance(computed, (int, float)) and pd.notna(computed) and pd.notna(mrd):
            xs.append(float(computed))
            ys.append(float(mrd))

    if not xs:
        raise RuntimeError("未收集到有效的 all_computed_tokens 与 model_run_duration_ms 配对数据")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, alpha=0.5, s=12, edgecolors="none")
    plt.xlabel("all_computed_tokens")
    plt.ylabel("model_run_duration_ms (ms)")
    plt.title("Model Run Duration vs All Computed Tokens (total_scheduled_tokens == 1)")
    plt.grid(True, alpha=0.3)

    # 拟合一条简单的线性趋势线（可选）
    if len(xs) > 2:
        try:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(xs), max(xs), 200)
            plt.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label="Trend")
            plt.legend()
        except Exception:
            pass

    out_path = Path("model_run_vs_all_computed_tokens.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✅ 已保存图表: {out_path.resolve()}")
    print(f"样本点数量: {len(xs)}（来自 {len(df)} 条批次记录，输入文件数 {len(jsonl_paths)}）")


if __name__ == "__main__":
    main()


