#!/usr/bin/env python3
"""计算 lmetric 五行(fig6)全部指标 -> JSON。
flowgpt 两行: online_replay CSV(send_time/total_time/ttft/tpot[=decode时长s])
其余三行: slo 客户端 mapped.csv(launch_time/finish_time/generation_time/generation_tokens)
输出: {row: {"x": [...], "p50": [...], "p90": [...], "ttft": [...], "tpot": [...], "slo": [...]}}
"""
import glob, json, os
import pandas as pd

BASE = "/home/ubuntu/zhangy/vllm-workspace/vllm/paper_figs/e2e_exp/data"
out = {}

def flowgpt_point(d, slo_s):
    fs = glob.glob(os.path.join(d, "*.csv"))
    if not fs:
        return None
    parts = []
    for f in fs:  # csv_process 语义:逐文件砍头尾再合并
        x = pd.read_csv(f)
        x["lat"] = x["total_time"] - x["send_time"]
        mn, mx = x["send_time"].min(), x["send_time"].max()
        parts.append(x[(x["send_time"] > mn + 30) & (x["send_time"] < mx - 30)])
    dd = pd.concat(parts, ignore_index=True)
    return dict(
        p50=dd["lat"].quantile(.5), p90=dd["lat"].quantile(.9),
        ttft=dd["ttft"].quantile(.5), tpot=dd["tpot"].quantile(.5),
        slo=(dd["lat"] <= slo_s).mean() * 100)

def slo_point(d, slo_s):
    fs = glob.glob(os.path.join(d, "mapped.csv"))
    if not fs:
        return None
    df = pd.read_csv(fs[0])
    df["lat"] = df["finish_time"] - df["launch_time"]
    mn, mx = df["launch_time"].min(), df["launch_time"].max()
    dd = df[(df["launch_time"] > mn + 30) & (df["launch_time"] < mx - 30)]
    tpot = (dd["generation_time"] / dd["generation_tokens"] * 1000)
    return dict(
        p50=dd["lat"].quantile(.5), p90=dd["lat"].quantile(.9),
        ttft=dd["ttft"].quantile(.5), tpot=tpot.quantile(.5),
        slo=(dd["lat"] <= slo_s).mean() * 100)

rows = [
    ("flowgpt_qps", [14, 15, 16, 17, 18], 10,
     lambda q: f"{BASE}/flowgpt_qps/native_lmetric/flowgpt_qps_native_lmetric_{q}", flowgpt_point),
    ("flowgpt_timestamp", [14, 15, 16, 17, 18], 20,
     lambda q: f"{BASE}/flowgpt_timestamp/native_lmetric/flowgpt_timestamp_native_lmetric_0.0_0.{2*q}", flowgpt_point),
    ("sharegpt", [40, 44, 48, 52], 6,
     lambda q: f"{BASE}/sharegpt/native_lmetric/sharegpt_native_lmetric_{q}", slo_point),
    ("arxiv", [12, 13, 14, 15, 16], 15,
     lambda q: f"{BASE}/arxiv/native_lmetric/arxiv_native_lmetric_{q}", slo_point),
    ("reasoning", [50, 52, 54, 56, 58], 6,
     lambda q: f"{BASE}/reasoning/native_lmetric/reasoning_native_lmetric_{q}", slo_point),
]

for row, xs, slo_s, pathf, pointf in rows:
    rec = {"x": [], "p50": [], "p90": [], "ttft": [], "tpot": [], "slo": []}
    for q in xs:
        m = pointf(pathf(q), slo_s)
        if m is None:
            print(f"MISSING {row} {q} -> {pathf(q)}")
            continue
        rec["x"].append(q)
        for k in ("p50", "p90", "ttft", "tpot", "slo"):
            rec[k].append(round(float(m[k]), 3))
    out[row] = rec

with open("/tmp/lmetric_metrics.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
