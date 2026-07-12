#!/bin/bash
# =============================================================================
# Figure 6 / ShareGPT 行(qps 模式),单条调度线的 QPS sweep。
# 用 online_replay_dataset.py + sharegpt.jsonl(chat 端点)。目标 qps Q 拆成
# ceil(Q/CHUNK) 个进程,各跑 jsonl 的一个 sample 切片([i/n,(i+1)/n]),错开 10s。
#
# 注:dataset 客户端 CSV 列是 send_time/total_time/ttft/tpot,而 csv_process 对
# 非-flowgpt 数据集要 launch_time/finish_time/generation_time/generation_tokens,
# 故每点跑完做一次列映射后处理,产出可被 draw.py 直接聚合的 CSV。
#
# 用法:
#   CONFIG=native_lmetric QPS_LIST="40 44 48 52" DUR=300 bash run_fig6_sharegpt.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/ubuntu/zhangy/venv/bin/python3}"
JSONL="${JSONL:-$SCRIPT_DIR/data/workloads/sharegpt.jsonl}"
API_BASE="${API_BASE:-http://localhost:8888/v1}"
MODEL="${MODEL:-Qwen/QwQ-32B}"
APIKEY="$(printf 'a%.0s' {1..32})"
DUR="${DUR:-300}"
E2E_SLO="${E2E_SLO:-6}"           # sharegpt 的 e2e SLO(见 draw.py)
CHUNK="${CHUNK:-8}"               # 每进程 qps 上限
CONFIG="${CONFIG:-native_lmetric}"
QPS_LIST="${QPS_LIST:-40 44 48 52}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/../../paper_figs/e2e_exp/data/sharegpt/$CONFIG}"

ts(){ date '+%F %T'; }

# 每个点前重启 router(约 6s),避免 lmetric 在持续高负载下被卡死影响后续点。
restart_router() {
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ":8888 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  sleep 2
  ( export PATH=/home/ubuntu/zhangy/venv/bin:$PATH
    setsid env PRODUCTION_STACK=/home/ubuntu/zhangy/production-stack MODEL="$MODEL" \
      NUM_ENGINES=8 ROUTING_LOGIC=lmetric VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true \
      bash "$SCRIPT_DIR/launch_router.sh" </dev/null >/tmp/router_perpoint.out 2>&1 & disown )
  local i
  for i in $(seq 1 20); do
    sleep 2
    [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)" = "200" ] && { echo "[$(ts)]  router restarted OK"; return 0; }
  done
  echo "[$(ts)]  WARN: router not healthy after restart"; return 1
}

echo "[$(ts)] ShareGPT sweep: config=$CONFIG qps=[$QPS_LIST] chunk=$CHUNK dur=${DUR}s -> $OUTROOT"

postprocess() {  # 给目录下每个 detailed CSV 加 csv_process 非-flowgpt 分支需要的列
  local dir="$1"
  "$VENV" - "$dir" <<'PY'
import sys, glob, pandas as pd
d = sys.argv[1]
for f in glob.glob(d + "/*.csv"):
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if "launch_time" in df.columns:
        continue
    df["launch_time"] = df["send_time"]
    df["finish_time"] = df["total_time"]                 # 绝对结束时刻
    ttft_time = df["send_time"] + df["ttft"]
    df["generation_time"] = (df["total_time"] - ttft_time).clip(lower=0)  # decode 时长(s)
    df["generation_tokens"] = df["tokens_out"]
    df.to_csv(f, index=False)
print("postprocessed:", d)
PY
}

for Q in $QPS_LIST; do
  restart_router
  OUT="$OUTROOT/sharegpt_${CONFIG}_${Q}"
  mkdir -p "$OUT"
  n=$(( (Q + CHUNK - 1) / CHUNK ))
  echo "[$(ts)]  == point Q=$Q -> $n procs (chunk<=$CHUNK) -> $OUT =="
  pids=()
  rem=$Q
  for ((i=0; i<n; i++)); do
    chunk=$CHUNK; [ "$rem" -lt "$CHUNK" ] && chunk=$rem
    s=$(awk -v i="$i" -v n="$n" 'BEGIN{printf "%.3f", i/n}')
    e=$(awk -v i="$i" -v n="$n" 'BEGIN{printf "%.3f", (i+1)/n}')
    tag=$(printf "qps%d_i%d_s%s_e%s" "$chunk" "$((i+1))" "$s" "$e")
    "$VENV" "$SCRIPT_DIR/online_replay_dataset.py" \
      --dataset "$JSONL" --input "$JSONL" --replay-mode qps --target-qps "$chunk" \
      --sample-range "$s" "$e" --use-chat true \
      --api-base "$API_BASE" --model "$MODEL" --api-key "$APIKEY" \
      --round-duration "$DUR" --max-rounds 1 --preload-time 20 \
      --e2e-slo "$E2E_SLO" --ttft-slo 1000 --tpot-slo 50 \
      --json-output "$OUT/sharegpt_${CONFIG}_${tag}.json" \
      --detailed-logs "$OUT/sharegpt_${CONFIG}_${tag}.csv" \
      > "$OUT/sharegpt_${CONFIG}_${tag}.log" 2>&1 &
    pids+=($!)
    rem=$((rem-chunk))
    [ "$i" -lt "$((n-1))" ] && sleep 10
  done
  wait "${pids[@]}"
  postprocess "$OUT"
  rows=$(cat "$OUT"/*.csv 2>/dev/null | grep -vc "^request_id")
  echo "[$(ts)]  point Q=$Q done, total csv rows ~$rows"
done
echo "[$(ts)] ALL_DONE config=$CONFIG"
