#!/bin/bash
# =============================================================================
# Figure 6 / FlowGPT-T 行,单条调度线的 sample-range sweep。
# 引擎已在跑(每卡 1 副本 QwQ-32B FP8,SLA off = sarathi);router 已在 :8888。
# 每个"点" = sample-range [0.0, U],按 0.1 切成多个客户端进程(与已有数据一致),
# 相邻进程错开 10s;产出 --detailed-logs CSV,供 paper_figs/e2e_exp draw.py 聚合。
#
#   x 轴 qps = 50 * U(0.28->14, 0.30->15, 0.32->16, 0.34->17, 0.36->18)
#
# 用法:
#   CONFIG=native_lmetric UPPERS="0.28 0.30 0.32 0.34 0.36" DUR=300 bash run_fig6_flowgpt_t.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/ubuntu/zhangy/venv/bin/python3}"
INPUT="${INPUT:-/tmp/flowgpt-head.log}"
API_BASE="${API_BASE:-http://localhost:8888/v1}"
MODEL="${MODEL:-Qwen/QwQ-32B}"
APIKEY="$(printf 'a%.0s' {1..32})"
DUR="${DUR:-300}"
E2E_SLO="${E2E_SLO:-20}"          # flowgpt_timestamp 的 e2e SLO(见 draw.py)
CONFIG="${CONFIG:-native_lmetric}"
UPPERS="${UPPERS:-0.28 0.30 0.32 0.34 0.36}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/../../paper_figs/e2e_exp/data/flowgpt_timestamp/$CONFIG}"

ts(){ date '+%F %T'; }
echo "[$(ts)] FlowGPT-T sweep: config=$CONFIG uppers=[$UPPERS] dur=${DUR}s -> $OUTROOT"

for U in $UPPERS; do
  Uh=$(awk -v u="$U" 'BEGIN{printf "%d", u*100+0.5}')
  OUT="$OUTROOT/flowgpt_timestamp_${CONFIG}_0.0_${U}"
  mkdir -p "$OUT"
  echo "[$(ts)]  == point U=$U (qps~$((Uh/2))) -> $OUT =="
  pids=()
  sh=0
  while [ "$sh" -lt "$Uh" ]; do
    eh=$((sh+10)); [ "$eh" -gt "$Uh" ] && eh=$Uh
    s=$(awk -v x="$sh" 'BEGIN{printf "%.2f", x/100}')
    e=$(awk -v x="$eh" 'BEGIN{printf "%.2f", x/100}')
    tag=$(printf "s%s_%s" "$s" "$e")
    "$VENV" "$SCRIPT_DIR/online_replay.py" \
      --input "$INPUT" --api-base "$API_BASE" --model "$MODEL" --api-key "$APIKEY" \
      --max-tokens 200 --replay-mode timestamp --sample-range "$s" "$e" \
      --round-duration "$DUR" --max-rounds 1 --preload-time 20 \
      --e2e-slo "$E2E_SLO" --ttft-slo 1000 --tpot-slo 50 \
      --json-output "$OUT/flowgpt_timestamp_${CONFIG}_${tag}.json" \
      --detailed-logs "$OUT/flowgpt_timestamp_${CONFIG}_${tag}.csv" \
      > "$OUT/flowgpt_timestamp_${CONFIG}_${tag}.log" 2>&1 &
    pids+=($!)
    sh=$((sh+10))
    [ "$sh" -lt "$Uh" ] && sleep 10
  done
  wait "${pids[@]}"
  rows=$(cat "$OUT"/*.csv 2>/dev/null | grep -vc "^request_id")
  echo "[$(ts)]  point U=$U done, total csv rows ~$rows"
done
echo "[$(ts)] ALL_DONE config=$CONFIG"
