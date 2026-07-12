#!/usr/bin/env python3
"""探针结果分析:对一个 probe 输出根目录(含 flowgpt_qps_<cfg>_<Q>/ 子目录)
按 csv_process 同款逻辑(latency=total_time-send_time,砍头尾)算指标,一行一个点。
用法: analyze_probe.py <outroot> [--trim 20]
"""
import sys, glob, os, re
import pandas as pd

root = sys.argv[1]
trim = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

rows = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(d):
        continue
    fs = glob.glob(os.path.join(d, "*.csv"))
    if not fs:
        rows.append((os.path.basename(d), None))
        continue
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["latency"] = df["total_time"] - df["send_time"]
    mn, mx = df["send_time"].min(), df["send_time"].max()
    dd = df[(df["send_time"] > mn + trim) & (df["send_time"] < mx - trim)]
    if len(dd) == 0:
        rows.append((os.path.basename(d), None))
        continue
    span = dd["send_time"].max() - dd["send_time"].min()
    m = re.search(r"_(\d+)$", os.path.basename(d))
    q = m.group(1) if m else "?"
    rows.append((q, dict(
        n=len(dd),
        aqps=len(dd) / span if span > 0 else 0,
        p50=dd["latency"].quantile(.5),
        p90=dd["latency"].quantile(.9),
        p99=dd["latency"].quantile(.99),
        slo10=(dd["latency"] <= 10).mean() * 100,
        ttft=dd["ttft"].quantile(.5),
        ttft90=dd["ttft"].quantile(.9),
        tpot=dd["tpot"].quantile(.5) * 1000 if dd["tpot"].max() < 10 else dd["tpot"].quantile(.5),
    )))

print("%-6s %-5s %-6s %-7s %-7s %-7s %-7s %-7s %-8s %-7s" % (
    "point", "n", "aqps", "p50", "p90", "p99", "SLO10", "ttft", "ttft90", "tpot"))
for q, m in rows:
    if m is None:
        print("%-6s NO DATA" % q)
    else:
        print("%-6s %-5d %-6.2f %-7.2f %-7.2f %-7.2f %-7.1f %-7.2f %-8.2f %-7.1f" % (
            q, m["n"], m["aqps"], m["p50"], m["p90"], m["p99"], m["slo10"],
            m["ttft"], m["ttft90"], m["tpot"]))
