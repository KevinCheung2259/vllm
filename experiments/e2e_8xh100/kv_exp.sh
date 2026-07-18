#!/bin/bash
# =============================================================================
# KV-cache 命中率对比实验: FlowGPT-T @ ~14qps × 4 路由策略
#   roundrobin / session / lmetric / prism(=SynergySched: LENS+ELRAR)
# 每策略重启全部引擎(冷缓存),300s,采集逐请求 cached_tokens。
# 产出: ~/kv_exp/<policy>/*.csv
# =============================================================================
set -uo pipefail
EXP=~/zhangy/vllm-workspace/vllm/experiments/e2e_8xh100
VENVPY=/home/ubuntu/zhangy/venv/bin/python3
OUTBASE=~/kv_exp
U="${U:-0.28}"
DUR="${DUR:-300}"
mkdir -p "$OUTBASE"
ts(){ date '+%F %T'; }

stop_all() {
  for i in $(seq 0 7); do docker rm -f e2e_engine_$i >/dev/null 2>&1; done
  pkill -f vllm_router.app 2>/dev/null
  sleep 3
}

wait_engines() {
  echo "[$(ts)] 等待 8 引擎就绪..."
  for i in $(seq 0 7); do
    port=$((8000+i))
    for t in $(seq 1 120); do
      code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:$port/health 2>/dev/null)
      [ "$code" = "200" ] && break
      sleep 5
    done
    [ "$code" = "200" ] && echo "  engine $i OK" || { echo "  engine $i FAILED"; docker logs e2e_engine_$i 2>&1 | tail -5; return 1; }
  done
}

wait_router() {
  for t in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)
    [ "$code" = "200" ] && { echo "[$(ts)] router OK"; return 0; }
    sleep 2
  done
  echo "router FAILED"; tail -5 $EXP/logs/router.log; return 1
}

run_policy() {
  local policy=$1 sla=$2 elrar=$3 logic=$4 skey=$5
  echo "[$(ts)] ================ POLICY=$policy ================"
  stop_all
  # 引擎(冷缓存)
  NUM_ENGINES=8 MODEL=Qwen/QwQ-32B QUANT=fp8 \
    SLA_ENABLED=$sla ELRAR_ENABLED=$elrar \
    bash $EXP/launch_engines_docker.sh > $OUTBASE/${policy}_engines.log 2>&1
  wait_engines || return 1
  # 路由
  local extra_env=""
  if [ "$policy" = "lmetric" ]; then
    export VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true
  fi
  ROUTING_LOGIC=$logic SESSION_KEY=$skey PYTHONPATH="" \
    bash -c "cd $EXP && PATH=/home/ubuntu/zhangy/venv/bin:\$PATH bash launch_router.sh" \
    > $OUTBASE/${policy}_router.log 2>&1
  wait_router || return 1
  # 客户端(单点 U, 按 0.1 切分并行进程, 与 fig6 协议一致)
  local OUT=$OUTBASE/$policy
  mkdir -p "$OUT"
  local Uh pids sh eh s e tag
  Uh=$(awk -v u="$U" 'BEGIN{printf "%d", u*100+0.5}')
  pids=(); sh=0
  while [ "$sh" -lt "$Uh" ]; do
    eh=$((sh+10)); [ "$eh" -gt "$Uh" ] && eh=$Uh
    s=$(awk -v x="$sh" 'BEGIN{printf "%.2f", x/100}')
    e=$(awk -v x="$eh" 'BEGIN{printf "%.2f", x/100}')
    tag=$(printf "s%s_%s" "$s" "$e")
    "$VENVPY" $EXP/online_replay.py \
      --input /tmp/flowgpt-head.log --api-base http://localhost:8888/v1 \
      --model Qwen/QwQ-32B --api-key aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
      --max-tokens 200 --replay-mode timestamp --sample-range "$s" "$e" \
      --round-duration "$DUR" --max-rounds 1 --preload-time 20 \
      --e2e-slo 20 --ttft-slo 1000 --tpot-slo 50 \
      --json-output "$OUT/${policy}_${tag}.json" \
      --detailed-logs "$OUT/${policy}_${tag}.csv" \
      > "$OUT/${policy}_${tag}.log" 2>&1 &
    pids+=($!)
    sh=$((sh+10))
    [ "$sh" -lt "$Uh" ] && sleep 10
  done
  wait "${pids[@]}"
  rows=$(cat "$OUT"/*.csv 2>/dev/null | grep -vc "^request_id")
  echo "[$(ts)] $policy done, csv rows=$rows"
}

case "${1:-all}" in
  roundrobin) run_policy roundrobin false false roundrobin x-user-id ;;
  session)    run_policy session    false false session    X-Flow-Conversation-Id ;;
  lmetric)    run_policy lmetric    false false lmetric    x-user-id ;;
  prism)      run_policy prism      true  true  elrar      X-Flow-Conversation-Id ;;
  all)
    run_policy roundrobin false false roundrobin x-user-id
    run_policy session    false false session    X-Flow-Conversation-Id
    run_policy lmetric    false false lmetric    x-user-id
    run_policy prism      true  true  elrar      X-Flow-Conversation-Id
    stop_all
    echo "[$(ts)] KV_EXP_ALL_DONE"
    ;;
esac
