#!/usr/bin/env python3
"""把 FlowGPT 生产日志 replay-logs-origin.log 转成统一 JSONL(复用 online_replay.py 的解析)。
每行: {"messages": [{role,content}...], "max_tokens": N}。取前 --limit 条有效请求。"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_replay import extract_json_from_log  # 复用已验证的日志解析

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="data/replay-logs-origin.log")
ap.add_argument("--out", default="data/workloads/flowgpt.jsonl")
ap.add_argument("--limit", type=int, default=3000)
ap.add_argument("--max-tokens", type=int, default=200)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
n = 0
with open(a.input, errors="ignore") as fin, open(a.out, "w") as fout:
    for line in fin:
        rd = None
        try: rd = extract_json_from_log(line)
        except Exception: rd = None
        if not rd: continue
        p = (rd.get("body") or {}).get("prompt")
        if isinstance(p, str) and p.strip():
            norm = [{"role": "user", "content": p}]          # FlowGPT: prompt 是字符串(已含模板)
        elif isinstance(p, list) and p:
            norm = [{"role": m.get("role","user"), "content": m.get("content","")} for m in p
                    if isinstance(m, dict) and m.get("content")]
        else:
            continue
        if not norm: continue
        fout.write(json.dumps({"messages": norm, "max_tokens": a.max_tokens}, ensure_ascii=False) + "\n")
        n += 1
        if a.limit and n >= a.limit: break
print(f"[flowgpt] -> {a.out} ({n} 条)")
