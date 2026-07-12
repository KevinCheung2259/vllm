#!/bin/bash
# =============================================================================
# QPS 扫点(单个 5 分钟窗口/点)。穿 router(:8888)打压,每点前后抓各引擎
# prefix-cache 计数用于算 KV cache 命中率。产出:
#   exp_dataset/<workload>/<workload>_qps<Q>.csv      逐请求明细(含 cached_tokens 列)
#   exp_dataset/<workload>/<workload>_qps<Q>.json     每轮聚合(客户端自带)
#   exp_dataset/<workload>/metrics/*_qps<Q>_{before,after}.txt  prefix-cache 快照
#
# 用法:
#   bash run_sweep.sh sharegpt                       # 默认扫 4 8 12 16
#   QPS_LIST="4 8 12 16 20 24" bash run_sweep.sh arxiv
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKLOAD="${1:-sharegpt}"
JSONL="${SCRIPT_DIR}/data/workloads/${WORKLOAD}.jsonl"
QPS_LIST="${QPS_LIST:-4 8 12 16}"

API_BASE="${API_BASE:-http://localhost:8888/v1}"     # 穿 router
MODEL="${MODEL:-Qwen/QwQ-32B}"
API_KEY="${API_KEY:-$(printf 'a%.0s' {1..32})}"
ROUND_DURATION="${ROUND_DURATION:-300}"              # 单个 5 分钟窗口
MAX_ROUNDS="${MAX_ROUNDS:-1}"
PRELOAD="${PRELOAD:-20}"
NUM_ENGINES="${NUM_ENGINES:-8}"
PORT_BASE="${PORT_BASE:-8000}"
PY="${PY:-/home/ubuntu/zhangy/venv/bin/python}"

CONFIG="${CONFIG:-synergysched}"   # 配置标签,区分不同方案的结果(synergysched / rr_baseline / ...)
OUT="${SCRIPT_DIR}/exp_dataset/${WORKLOAD}_${CONFIG}"
mkdir -p "$OUT/metrics"

[ -f "$JSONL" ] || { echo "缺少 $JSONL,先跑 prepare_workload.py"; exit 1; }

snapshot() {  # $1=qps $2=tag(before/after)
  local out="$OUT/metrics/${WORKLOAD}_qps$1_$2.txt"
  : > "$out"
  for ((e=0; e<NUM_ENGINES; e++)); do
    local p=$((PORT_BASE + e))
    echo "# engine $e :$p" >> "$out"
    curl -s "localhost:$p/metrics" 2>/dev/null | grep -E "vllm:gpu_prefix_cache_(hits|queries)_total" | grep -v '^#' >> "$out"
  done
}

echo "workload=$WORKLOAD  QPS 点: $QPS_LIST  每点 ${ROUND_DURATION}s×${MAX_ROUNDS}  经 $API_BASE"
for q in $QPS_LIST; do
  echo "===== QPS=$q 开始 $(date '+%T') ====="
  snapshot "$q" before
  "$PY" online_replay_dataset.py \
    --dataset "$JSONL" --input "$JSONL" \
    --replay-mode qps --target-qps "$q" \
    --api-base "$API_BASE" --model "$MODEL" --api-key "$API_KEY" \
    --use-chat true --round-duration "$ROUND_DURATION" --max-rounds "$MAX_ROUNDS" \
    --preload-time "$PRELOAD" \
    --json-output "$OUT/${WORKLOAD}_qps${q}.json" \
    --detailed-logs "$OUT/${WORKLOAD}_qps${q}.csv" 2>&1 | grep -viE "^INFO 07|HTTP/1.1|warnings.warn|Skipping" | tail -3
  snapshot "$q" after
  echo "===== QPS=$q 完成 $(date '+%T'),明细: $OUT/${WORKLOAD}_qps${q}.csv ====="
  sleep 5
done
echo "全部 QPS 点完成。"
