#!/bin/bash
# =============================================================================
# 公开数据集(ShareGPT / Code-Reasoning / arXiv-Summarization)的客户端压测脚本
# —— 对应论文中除 FlowGPT 外的三个 workload,统一走 QPS 模式。
#
# 前置:
#   1. 引擎 + router 已起(router 在 :8888,模型名与 --model 一致);
#   2. 已用 prepare_workload.py 生成 data/workloads/<workload>.jsonl
#      (本脚本会在缺失时自动尝试生成)。
#
# 用法:
#   bash run_dataset_workload.sh sharegpt
#   MODEL=Qwen/QwQ-32B QPS=10 bash run_dataset_workload.sh arxiv
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKLOAD="${1:-sharegpt}"                      # sharegpt | code | arxiv
JSONL="${SCRIPT_DIR}/data/workloads/${WORKLOAD}.jsonl"

# ------------------------------- 可调配置 -----------------------------------
API_BASE="${API_BASE:-http://localhost:8888/v1}"
MODEL="${MODEL:-Qwen/QwQ-32B}"
API_KEY="${API_KEY:-$(printf 'a%.0s' {1..32})}"
QPS="${QPS:-10}"
LIMIT="${LIMIT:-0}"                            # 转换时每个 workload 最多多少条(0=全部)
MAX_ROUNDS="${MAX_ROUNDS:-1}"
# 论文按场景设 SLO:交互类严 TPOT,摘要类严 TTFT。这里给每个 workload 合理默认。
case "$WORKLOAD" in
  sharegpt) E2E_SLO="${E2E_SLO:-5}";  TTFT_SLO="${TTFT_SLO:-1000}"; TPOT_SLO="${TPOT_SLO:-50}";;
  code)     E2E_SLO="${E2E_SLO:-10}"; TTFT_SLO="${TTFT_SLO:-1500}"; TPOT_SLO="${TPOT_SLO:-50}";;
  arxiv)    E2E_SLO="${E2E_SLO:-20}"; TTFT_SLO="${TTFT_SLO:-2000}"; TPOT_SLO="${TPOT_SLO:-80}";;
  *) echo "未知 workload: $WORKLOAD (可选 sharegpt|code|arxiv)"; exit 1;;
esac
# ---------------------------------------------------------------------------

# 数据集缺失则尝试生成
if [ ! -f "$JSONL" ]; then
  echo "[$WORKLOAD] JSONL 不存在,尝试用 prepare_workload.py 生成 ..."
  python3 prepare_workload.py --workload "$WORKLOAD" ${LIMIT:+--limit "$LIMIT"} || {
    echo "生成失败。若因 host pyarrow 版本问题,请在引擎容器内运行 prepare_workload.py。"; exit 1; }
fi

OUT_DIR="${SCRIPT_DIR}/exp_dataset/${WORKLOAD}"
mkdir -p "$OUT_DIR"
TAG="${WORKLOAD}_qps${QPS}"

echo "[$WORKLOAD] 压测: model=$MODEL api=$API_BASE qps=$QPS  SLO(e2e=${E2E_SLO}s ttft=${TTFT_SLO}ms tpot=${TPOT_SLO}ms)"
python3 online_replay_dataset.py \
  --dataset "$JSONL" \
  --input "$JSONL" \
  --replay-mode qps \
  --target-qps "$QPS" \
  --api-base "$API_BASE" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --use-chat true \
  --max-rounds "$MAX_ROUNDS" \
  --e2e-slo "$E2E_SLO" --ttft-slo "$TTFT_SLO" --tpot-slo "$TPOT_SLO" \
  --preload-time 15 \
  --json-output "${OUT_DIR}/${TAG}.json" \
  --detailed-logs "${OUT_DIR}/${TAG}.csv"

echo "[$WORKLOAD] 完成。结果:${OUT_DIR}/${TAG}.json"
