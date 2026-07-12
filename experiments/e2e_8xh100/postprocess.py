#!/usr/bin/env python3
"""
后处理 QPS 扫点结果:丢头尾各 30s 取稳态,算 E2E/TTFT/TPOT 百分位、吞吐、达成QPS、KV命中率。

CSV 列: request_id,conversation_id,send_time,ttft_time,total_time,tokens_in,tokens_out,ttft,tpot,cached_tokens
  - ttft(col)  : TTFT 秒
  - tpot(col)  : 解码总时长 = total_time - ttft(秒),非每token
  - E2E        : ttft + tpot
  - 真实 TPOT  : tpot / max(tokens_out-1,1) (ms/token)
KV命中率: 从 metrics/<wl>_qps<Q>_{before,after}.txt 求 Δhits/Δqueries(8 引擎汇总)
"""
import sys, os, glob, csv, re

WL = sys.argv[1] if len(sys.argv) > 1 else "sharegpt"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "synergysched"
QPS_LIST = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["4","8","12","16"])]
TRIM = 30.0  # 头尾各丢 30s
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "exp_dataset", f"{WL}_{CONFIG}")
print(f"[配置: {CONFIG}]  数据目录: {OUT}\n")

def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); k = (len(xs)-1)*p/100.0
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def kv_hit_rate(q):
    def parse(tag):
        fp = os.path.join(OUT, "metrics", f"{WL}_qps{q}_{tag}.txt")
        h = tot = 0
        if os.path.exists(fp):
            for ln in open(fp):
                m = re.search(r"gpu_prefix_cache_(hits|queries)_total\S*\s+([\d.eE+]+)", ln)
                if m:
                    v = float(m.group(2))
                    if m.group(1) == "hits": h += v
                    else: tot += v
        return h, tot
    hb, qb = parse("before"); ha, qa = parse("after")
    dh, dq = ha-hb, qa-qb
    return (dh/dq*100.0) if dq > 0 else float("nan")

DUR = 300.0  # 每个负载点的窗口时长(round_duration)
hdr = ("目标QPS","完成数","吞吐req/s","输出tok/s","E2E-P50s","E2E-P90s","TTFT-P50s","TPOTms/tok","KV命中%")
print("".join(f"{h:>11}" for h in hdr))
print("-"*99)
for q in QPS_LIST:
    fp = os.path.join(OUT, f"{WL}_qps{q}.csv")
    if not os.path.exists(fp):
        print(f"{q:>11}  (缺 {fp})"); continue
    rows = []
    with open(fp) as f:
        for r in csv.DictReader(f):
            try:
                ttft=float(r["ttft"]); tp=float(r["tpot"]); out=int(r["tokens_out"])
                if ttft>0 and tp>=0 and out>0: rows.append((ttft,tp,out))
            except: pass
    if not rows:
        print(f"{q:>11}  (无有效数据)"); continue
    e2e=[a+b for a,b,_ in rows]; ttfts=[a for a,_,_ in rows]
    tpot_ms=[b/max(c-1,1)*1000 for _,b,c in rows]
    thr=len(rows)/DUR; tok=sum(c for _,_,c in rows)/DUR
    vals=(q,len(rows),f"{thr:.2f}",f"{tok:.0f}",f"{pct(e2e,50):.1f}",f"{pct(e2e,90):.1f}",
          f"{pct(ttfts,50):.2f}",f"{pct(tpot_ms,50):.0f}",f"{kv_hit_rate(q):.1f}")
    print("".join(f"{str(v):>11}" for v in vals))
print("\n注:吞吐=完成数/300s(真实可持续);E2E/TTFT 秒;TPOT 每输出token毫秒;KV命中=跑中Δhits/Δqueries。")
