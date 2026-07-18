#!/bin/bash
# KV 实验 v2: 每策略 = 引擎冷启 -> 暖机(不相交切片 0.40-0.68, 360s) -> 正式测量(0.0-0.28, 600s)
set -uo pipefail
EXP=~/zhangy/vllm-workspace/vllm/experiments/e2e_8xh100
VENVPY=/home/ubuntu/zhangy/venv/bin/python3
OUTBASE=~/kv_exp2
DUR=600
WARM_DUR=360
mkdir -p "$OUTBASE"
ts(){ date '+%F %T'; }

stop_all(){ for i in $(seq 0 7); do docker rm -f e2e_engine_$i >/dev/null 2>&1; done; pkill -f vllm_router.app 2>/dev/null; sleep 3; }

wait_engines(){
  for i in $(seq 0 7); do
    port=$((8000+i))
    for t in $(seq 1 120); do
      code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:$port/health 2>/dev/null)
      [ "$code" = "200" ] && break; sleep 5
    done
    [ "$code" = "200" ] || { echo "engine $i FAILED"; return 1; }
  done; echo "[$(ts)] engines OK"
}
wait_router(){
  for t in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)
    [ "$code" = "200" ] && { echo "[$(ts)] router OK"; return 0; }; sleep 2
  done; echo "router FAILED"; return 1
}

client(){  # $1=sample_start $2=sample_end $3=dur $4=outprefix
  "$VENVPY" $EXP/online_replay.py \
    --input /tmp/flowgpt-head.log --api-base http://localhost:8888/v1 \
    --model Qwen/QwQ-32B --api-key aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
    --max-tokens 200 --replay-mode timestamp --sample-range "$1" "$2" \
    --round-duration "$3" --max-rounds 1 --preload-time 20 \
    --e2e-slo 20 --ttft-slo 1000 --tpot-slo 50 \
    --json-output "$4.json" --detailed-logs "$4.csv" > "$4.log" 2>&1
}

run_policy(){
  local policy=$1 sla=$2 elrar=$3 logic=$4 skey=$5
  echo "[$(ts)] ========== POLICY=$policy (warmup+measure) =========="
  stop_all
  NUM_ENGINES=8 MODEL=Qwen/QwQ-32B QUANT=fp8 SLA_ENABLED=$sla ELRAR_ENABLED=$elrar \
    SLO_TPOT_MS=100 MIN_BATCH=32 MIN_BATCH_K=2.0 \
    bash $EXP/launch_engines_docker.sh > $OUTBASE/${policy}_engines.log 2>&1
  wait_engines || return 1
  [ "$policy" = "lmetric" ] && export VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true
  ROUTING_LOGIC=$logic SESSION_KEY=$skey PYTHONPATH="" \
    bash -c "cd $EXP && PATH=/home/ubuntu/zhangy/venv/bin:\$PATH bash launch_router.sh" \
    > $OUTBASE/${policy}_router.log 2>&1
  wait_router || return 1
  local OUT=$OUTBASE/$policy; mkdir -p "$OUT"
  # 暖机: 不相交切片, 3 并行进程
  echo "[$(ts)] $policy warmup ${WARM_DUR}s"
  wp=()
  client 0.40 0.50 $WARM_DUR "$OUT/warm_1" & wp+=($!)
  sleep 5; client 0.50 0.60 $WARM_DUR "$OUT/warm_2" & wp+=($!)
  sleep 5; client 0.60 0.68 $WARM_DUR "$OUT/warm_3" & wp+=($!)
  wait "${wp[@]}"
  echo "[$(ts)] $policy warmup done, measuring ${DUR}s"
  # 正式测量
  mp=()
  client 0.00 0.10 $DUR "$OUT/meas_1" & mp+=($!)
  sleep 10; client 0.10 0.20 $DUR "$OUT/meas_2" & mp+=($!)
  sleep 10; client 0.20 0.28 $DUR "$OUT/meas_3" & mp+=($!)
  wait "${mp[@]}"
  rows=$(cat "$OUT"/meas_*.csv 2>/dev/null | grep -vc "^request_id")
  echo "[$(ts)] $policy done, meas rows=$rows"
}

run_policy roundrobin false false roundrobin x-user-id
run_policy session    false false session    X-Flow-Conversation-Id
run_policy lmetric    false false lmetric    x-user-id
run_policy prism      true  true  elrar      X-Flow-Conversation-Id
stop_all
echo "[$(ts)] KV_EXP2_ALL_DONE"
