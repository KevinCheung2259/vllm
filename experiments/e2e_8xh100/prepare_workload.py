#!/usr/bin/env python3
"""
把论文 8×H100 实验用的异构数据集统一转成 replay 客户端可消费的 JSONL。

输出每行一个请求,格式与 online_replay 的 ShareGPT 解析分支一致:
    {"conversations": [{"from": "human", "value": <prompt>}], "max_tokens": <int>}

支持的 workload(本地文件已备好在 experiments/e2e_8xh100/data/ 下):
  - sharegpt : ShareGPT_Vicuna_unfiltered (对话)          -> 取首个 human turn
  - code     : GetSoloTech/Code-Reasoning (代码)          -> question 字段
  - arxiv    : whu9/arxiv_summarization_postprocess (摘要) -> source 字段 + 摘要指令

用法:
  python3 prepare_workload.py --workload arxiv --limit 500
  python3 prepare_workload.py --workload all --limit 1000        # 全部转,每个最多 1000 条
输出:data/workloads/<workload>.jsonl
"""
import argparse
import json
import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT_DIR = os.path.join(DATA, "workloads")

# 本地数据集路径
SHAREGPT_JSON = os.environ.get(
    "SHAREGPT_JSON",
    "/home/ubuntu/qq/llm-inference-benchmarking/ShareGPT_V3_unfiltered_cleaned_split.json",
)
CODE_DIR = os.path.join(DATA, "Code-Reasoning", "data")
ARXIV_DIR = os.path.join(DATA, "arxiv_summarization_postprocess", "data")

# 各 workload 默认输出 token 上限(参考论文 Tab.1 的 Output Tokens 量级)
DEFAULT_MAX_TOKENS = {"sharegpt": 256, "code": 512, "arxiv": 300}


def _emit(fout, prompt, max_tokens):
    if not prompt or not prompt.strip():
        return 0
    rec = {
        "conversations": [{"from": "human", "value": prompt}],
        "max_tokens": max_tokens,
    }
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1


def convert_sharegpt(limit, max_tokens):
    out = os.path.join(OUT_DIR, "sharegpt.jsonl")
    n = 0
    with open(SHAREGPT_JSON) as f:
        data = json.load(f)
    with open(out, "w") as fout:
        for item in data:
            convs = item.get("conversations") or []
            # 取第一个 human/user turn 作为单轮 prompt
            prompt = next(
                (c.get("value") for c in convs
                 if c.get("from") in ("human", "user")),
                None,
            )
            n += _emit(fout, prompt, max_tokens)
            if limit and n >= limit:
                break
    return out, n


def _iter_parquet(pdir, columns):
    # 只读需要的列:避开 pyarrow 19 对 list 列 iter_batches 的 histogram bug,且更省内存
    import pyarrow.parquet as pq
    for fp in sorted(glob.glob(os.path.join(pdir, "*.parquet"))):
        tbl = pq.read_table(fp, columns=columns)
        col = tbl.column(columns[0]).to_pylist()
        for v in col:
            yield v


def convert_code(limit, max_tokens):
    out = os.path.join(OUT_DIR, "code.jsonl")
    n = 0
    with open(out, "w") as fout:
        for prompt in _iter_parquet(CODE_DIR, ["question"]):
            n += _emit(fout, prompt, max_tokens)
            if limit and n >= limit:
                break
    return out, n


def convert_arxiv(limit, max_tokens):
    out = os.path.join(OUT_DIR, "arxiv.jsonl")
    n = 0
    instr = "Summarize the following paper into a concise abstract:\n\n"
    with open(out, "w") as fout:
        for src in _iter_parquet(ARXIV_DIR, ["source"]):
            prompt = (instr + src) if src else None
            n += _emit(fout, prompt, max_tokens)
            if limit and n >= limit:
                break
    return out, n


CONVERTERS = {"sharegpt": convert_sharegpt, "code": convert_code, "arxiv": convert_arxiv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["sharegpt", "code", "arxiv", "all"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="每个 workload 最多转多少条(0=全部)")
    ap.add_argument("--max-tokens", type=int, default=0, help="覆盖输出 token 上限(0=用默认)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    targets = list(CONVERTERS) if args.workload == "all" else [args.workload]
    for w in targets:
        mt = args.max_tokens or DEFAULT_MAX_TOKENS[w]
        out, n = CONVERTERS[w](args.limit, mt)
        print(f"[{w}] -> {out}  ({n} 条, max_tokens={mt})")


if __name__ == "__main__":
    main()
