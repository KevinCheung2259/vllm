#!/bin/bash
# =============================================================================
# Figure 6 / ShareGPT 行 —— 用 production-stack 的 multi-round-qa benchmark
# (YNewsoul/production-stack benchmarks/multi-round-qa/multi-round-qa-fix.py)。
# 多轮对话:1000-token 共享系统前缀 + 每用户 20000-token 历史 + answer-len 100。
# 大部分输入是可缓存历史前缀 -> KV-aware 路由(lmetric)把同一用户路由到缓存其
# 历史的引擎,有效 prefill 很小 -> 高 qps 可行(朴素路由会崩)。这正是图要展示的。
# 输出 CSV 列(launch_time/finish_time/generation_time/generation_tokens/ttft)
# 直接匹配 e2e_exp/csv_process 的非-flowgpt 分支。
#
# 用法:
#   CONFIG=native_lmetric QPS_LIST="40 44 48 52" bash run_fig6_sharegpt_mrqa.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRQA="${MRQA:-/home/ubuntu/zhangy/mrqa}"
MRQA_SCRIPT="${MRQA_SCRIPT:-multi-round-qa.py}"
VENV="${VENV:-/home/ubuntu/zhangy/venv/bin/python3}"
DATA="${DATA:-$MRQA/sharegpt.json}"
BASE_URL="${BASE_URL:-http://localhost:8888/v1}"
MODEL="${MODEL:-Qwen/QwQ-32B}"
APIKEY="$(printf 'a%.0s' {1..32})"
CONFIG="${CONFIG:-native_lmetric}"
QPS_LIST="${QPS_LIST:-40 44 48 52}"
NUM_USERS="${NUM_USERS:-320}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
SYS_PROMPT="${SYS_PROMPT:-1000}"
HIST="${HIST:-20000}"
ANS="${ANS:-100}"
TIME_S="${TIME_S:-300}"
# multi-round-qa-fix.py 的 --round-data:循环取对话的模数,须 <= 数据集中满足
# num_round>=2*num_rounds 的对话数(sharegpt.json 里 >=20 轮的有 3098 条)。
ROUND_DATA="${ROUND_DATA:-1000}"
# USE_SHAREGPT=false(默认)= run.sh 的【合成】模式:prompt 由 shared-system-prompt
# (1000) + user-history-prompt(20000)合成 -> ~21k token 重负载,匹配论文/之前结果。
# =true 则用真实 sharegpt 对话原文(短 ~500 token,负载轻,latency 低,不匹配)。
USE_SHAREGPT="${USE_SHAREGPT:-false}"
if [ "$USE_SHAREGPT" = "true" ]; then SG_ARGS="--data-file $DATA --sharegpt"; else SG_ARGS=""; fi
OUTROOT="${OUTROOT:-$SCRIPT_DIR/../../paper_figs/e2e_exp/data/sharegpt/$CONFIG}"

ts(){ date '+%F %T'; }

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

warmup() {
  echo "[$(ts)] warmup (populate KV): num-users 400, time 200s ..."
  timeout 400 "$VENV" "$MRQA/$MRQA_SCRIPT" \
    --num-users 400 --num-rounds 2 --qps 4 --round-data "$ROUND_DATA" \
    --shared-system-prompt "$SYS_PROMPT" --user-history-prompt "$HIST" --answer-len "$ANS" \
    --model "$MODEL" --base-url "$BASE_URL" --api-key "$APIKEY" \
    $SG_ARGS --output /tmp/mrqa_warmup.csv \
    --log-interval 30 --time 200 > /tmp/mrqa_warmup.log 2>&1 || true
  echo "[$(ts)] warmup done"
}

echo "[$(ts)] ShareGPT(mrqa) sweep: config=$CONFIG qps=[$QPS_LIST] users=$NUM_USERS rounds=$NUM_ROUNDS sys=$SYS_PROMPT hist=$HIST ans=$ANS time=${TIME_S}s -> $OUTROOT"

restart_router
warmup

for Q in $QPS_LIST; do
  restart_router
  OUT="$OUTROOT/sharegpt_${CONFIG}_${Q}"
  mkdir -p "$OUT"
  echo "[$(ts)]  == point Q=$Q -> $OUT =="
  timeout $((TIME_S + 240)) "$VENV" "$MRQA/$MRQA_SCRIPT" \
    --num-users "$NUM_USERS" --num-rounds "$NUM_ROUNDS" --qps "$Q" --round-data "$ROUND_DATA" \
    --shared-system-prompt "$SYS_PROMPT" --user-history-prompt "$HIST" --answer-len "$ANS" \
    --model "$MODEL" --base-url "$BASE_URL" --api-key "$APIKEY" \
    $SG_ARGS --output "$OUT/sharegpt_${CONFIG}_${Q}.csv" \
    --log-interval 30 --time "$TIME_S" > "$OUT/sharegpt_${CONFIG}_${Q}.log" 2>&1 || true
  rows=$(cat "$OUT"/*.csv 2>/dev/null | grep -vc "launch_time")
  echo "[$(ts)]  point Q=$Q done, csv rows ~$rows"
done
echo "[$(ts)] ALL_DONE config=$CONFIG"
