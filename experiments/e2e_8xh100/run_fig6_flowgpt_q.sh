#!/bin/bash
# =============================================================================
# Figure 6 / FlowGPT-Q 行(qps 模式),单条调度线的 QPS sweep。
# 与已有数据一致:目标 qps Q 拆成 ceil(Q/5) 个进程,每进程 <=5 qps,各跑原始日志
# 的一个 sample 切片(切片总宽 = Q/50),相邻进程错开 10s。用 online_replay.py
# (qps 模式,读原始日志),产出 --detailed-logs CSV(与 flowgpt_timestamp 同 schema)。
#
# 用法:
#   CONFIG=native_lmetric QPS_LIST="14 15 16 17 18" DUR=300 bash run_fig6_flowgpt_q.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/home/ubuntu/zhangy/venv/bin/python3}"
INPUT="${INPUT:-/tmp/flowgpt-head.log}"
API_BASE="${API_BASE:-http://localhost:8888/v1}"
MODEL="${MODEL:-Qwen/QwQ-32B}"
APIKEY="$(printf 'a%.0s' {1..32})"
DUR="${DUR:-300}"
E2E_SLO="${E2E_SLO:-10}"          # flowgpt_qps 的 e2e SLO(见 draw.py)
CONFIG="${CONFIG:-native_lmetric}"
QPS_LIST="${QPS_LIST:-14 15 16 17 18}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/../../paper_figs/e2e_exp/data/flowgpt_qps/$CONFIG}"

ts(){ date '+%F %T'; }

# lmetric 路由在持续高负载下可能被卡死(socket 风暴/事件循环),故每个点前重启一次
# router(仅约 6s),保证每点都在健康 router 上跑。
restart_router() {
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ":8888 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  sleep 2
  ( export PATH=/home/ubuntu/zhangy/venv/bin:$PATH
    setsid env PRODUCTION_STACK=/home/ubuntu/zhangy/production-stack MODEL="$MODEL" \
      NUM_ENGINES=8 ROUTING_LOGIC="${ROUTING_LOGIC:-lmetric}" VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true \
      bash "$SCRIPT_DIR/launch_router.sh" </dev/null >/tmp/router_perpoint.out 2>&1 & disown )
  local i
  for i in $(seq 1 20); do
    sleep 2
    [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)" = "200" ] && { echo "[$(ts)]  router restarted OK"; return 0; }
  done
  echo "[$(ts)]  WARN: router not healthy after restart"; return 1
}

echo "[$(ts)] FlowGPT-Q sweep: config=$CONFIG qps=[$QPS_LIST] dur=${DUR}s -> $OUTROOT"

for Q in $QPS_LIST; do
  restart_router
  OUT="$OUTROOT/flowgpt_qps_${CONFIG}_${Q}"
  mkdir -p "$OUT"
  # ceil(Q/5) 个进程;切片总宽 upper=Q/50
  n=$(( (Q + 4) / 5 ))
  upper=$(awk -v q="$Q" 'BEGIN{printf "%.4f", q/50.0}')
  echo "[$(ts)]  == point Q=$Q -> $n procs, sample upper=$upper -> $OUT =="
  pids=()
  rem=$Q
  for ((i=0; i<n; i++)); do
    chunk=5; [ "$rem" -lt 5 ] && chunk=$rem
    s=$(awk -v i="$i"   -v u="$upper" -v n="$n" 'BEGIN{printf "%.3f", i*u/n}')
    e=$(awk -v i="$i"   -v u="$upper" -v n="$n" 'BEGIN{printf "%.3f", (i+1)*u/n}')
    tag=$(printf "qps%d_i%d_s%s_e%s" "$chunk" "$((i+1))" "$s" "$e")
    "$VENV" "$SCRIPT_DIR/online_replay.py" \
      --input "$INPUT" --api-base "$API_BASE" --model "$MODEL" --api-key "$APIKEY" \
      --max-tokens 200 --replay-mode qps --target-qps "$chunk" --sample-range "$s" "$e" \
      --round-duration "$DUR" --max-rounds 1 --preload-time 20 \
      --e2e-slo "$E2E_SLO" --ttft-slo 1000 --tpot-slo 50 \
      --json-output "$OUT/flowgpt_qps_${CONFIG}_${tag}.json" \
      --detailed-logs "$OUT/flowgpt_qps_${CONFIG}_${tag}.csv" \
      > "$OUT/flowgpt_qps_${CONFIG}_${tag}.log" 2>&1 &
    pids+=($!)
    rem=$((rem-chunk))
    [ "$i" -lt "$((n-1))" ] && sleep 10
  done
  wait "${pids[@]}"
  rows=$(cat "$OUT"/*.csv 2>/dev/null | grep -vc "^request_id")
  echo "[$(ts)]  point Q=$Q done, total csv rows ~$rows"
done
echo "[$(ts)] ALL_DONE config=$CONFIG"
