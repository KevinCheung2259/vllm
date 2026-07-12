#!/bin/bash
# =============================================================================
# 8×H100 引擎启动(容器版)—— 每卡 1 个 vLLM 实例,不用 PP/TP
#
# 环境 = 镜像 zhangy2259/vllm:2025-08-25-8xh100 (torch 2.7.0 + CUDA 12.8),
# 代码 = 运行时挂载 host 的 exp vllm 源码(含 sla_aware + engine_agent,
#        其 _C.abi3.so 正是为 torch 2.7 编译,故在容器内可正常加载)。
#
# 每个引擎:单 GPU、端口 8000+i、启用 SLA 调度器 + ELRAR Agent(UDP 单播到 gateway)。
# 用 --network host,便于引擎/gateway/router 在 localhost 互通。
#
# 用法:
#   NUM_ENGINES=1 MODEL=Qwen/Qwen2.5-0.5B-Instruct bash launch_engines_docker.sh   # 1 卡冒烟
#   NUM_ENGINES=8 MODEL=Qwen/QwQ-32B QUANT=fp8      bash launch_engines_docker.sh   # 8 卡复现
#   bash stop_all.sh                                                                # 停止
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "$LOG_DIR"

# ------------------------------- 可调配置 -----------------------------------
IMAGE="${IMAGE:-zhangy2259/vllm:2025-08-25-8xh100}"
VLLM_SRC="${VLLM_SRC:-/home/ubuntu/zhangy/vllm-workspace/vllm}"   # 挂载进容器的 exp vllm 源码
HF_CACHE="${HF_CACHE:-/mnt/local/hf-cache}"                       # 模型缓存放 2TB 盘,避免撑爆系统盘
MODEL="${MODEL:-Qwen/QwQ-32B}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"
QUANT="${QUANT:-}"                       # 置 fp8 则加 --quantization fp8(论文 QwQ-32B 用 FP8)
NUM_ENGINES="${NUM_ENGINES:-8}"
PORT_BASE="${PORT_BASE:-8000}"
GPU_BASE="${GPU_BASE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

SLA_ENABLED="${SLA_ENABLED:-true}"      # 关它=基线(vanilla 引擎调度)
ELRAR_ENABLED="${ELRAR_ENABLED:-true}"  # 关它=不推 ELRAR 状态(RR/session 路由用)
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-9999}"
SLO_TPOT_MS="${SLO_TPOT_MS:-50.0}"
SLO_TTFT_MS="${SLO_TTFT_MS:-1000.0}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/vllm-workspace/vllm/vllm/v1/core/sched/sla_aware/stable_model_h100_qwen32b.pkl}"
# ---------------------------------------------------------------------------

mkdir -p "$HF_CACHE"
QUANT_ARG=""; [ -n "$QUANT" ] && QUANT_ARG="--quantization $QUANT"
KV_DTYPE_ARG=""; [ -n "${KV_CACHE_DTYPE:-}" ] && KV_DTYPE_ARG="--kv-cache-dtype ${KV_CACHE_DTYPE}"

echo "镜像: $IMAGE"
echo "挂载 vllm 源码: $VLLM_SRC -> /vllm-workspace/vllm"
echo "模型: $MODEL  (量化: ${QUANT:-none})  引擎数: $NUM_ENGINES  Gateway: ${GATEWAY_HOST}:${GATEWAY_PORT}"

for ((i=0; i<NUM_ENGINES; i++)); do
  gpu=$((GPU_BASE + i)); port=$((PORT_BASE + i)); name="e2e_engine_${i}"
  engine_url="http://localhost:${port}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo "  [engine ${i}] GPU=${gpu} port=${port} container=${name}"

  docker run -d --name "$name" \
    --gpus "\"device=${gpu}\"" \
    --network host --ipc=host --shm-size=16g \
    -v "${VLLM_SRC}:/vllm-workspace/vllm" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e VLLM_SLA_SCHEDULER_ENABLED="${SLA_ENABLED}" \
    -e VLLM_SLA_FALLBACK_ON_ERROR=true \
    -e VLLM_SLO_TPOT_MS="${SLO_TPOT_MS}" \
    -e VLLM_SLO_TTFT_MS="${SLO_TTFT_MS}" \
    -e VLLM_SLA_USE_PRETRAINED=true \
    -e VLLM_SLA_PRETRAINED_PATH="${PRETRAINED_MODEL}" \
    -e VLLM_ENABLE_ELRAR="${ELRAR_ENABLED}" \
    -e VLLM_ELRAR_NETWORK_MODE=unicast \
    -e VLLM_ELRAR_GATEWAY_HOST="${GATEWAY_HOST}" \
    -e VLLM_ELRAR_GATEWAY_PORT="${GATEWAY_PORT}" \
    -e VLLM_ELRAR_ENGINE_ID="${engine_url}" \
    -e VLLM_SLA_EXPECTED_OUTPUT_LEN="${EXPECTED_OUTPUT_LEN:-256}" \
    -e VLLM_SLA_QUEUE_PENALTY="${QUEUE_PENALTY:-1.0}" \
    -e VLLM_SLA_MIN_BATCH="${MIN_BATCH:-1}" \
    -e VLLM_SLA_MIN_BATCH_K="${MIN_BATCH_K:-2.0}" \
    -e VLLM_ELRAR_PUSH_INTERVAL="${ELRAR_PUSH_INTERVAL:-200}" \
    -e VLLM_SLA_KNOB_FILE="${SLA_KNOB_FILE:-}" \
    --entrypoint bash \
    "$IMAGE" -lc "cd /vllm-workspace && exec vllm serve '${MODEL}' \
      --served-model-name '${SERVED_NAME}' \
      --port ${port} \
      --tensor-parallel-size 1 \
      --max-model-len ${MAX_MODEL_LEN} \
      --gpu-memory-utilization ${GPU_MEM_UTIL} \
      --max-num-seqs ${MAX_NUM_SEQS} \
      --enable-chunked-prefill \
      --disable-log-requests \
      ${QUANT_ARG} ${KV_DTYPE_ARG}" \
    > /dev/null

  echo "$name" > "${LOG_DIR}/engine_${i}.container"
  sleep 2
done

echo ""
echo "全部引擎容器已启动。观察日志/就绪:"
echo "  docker logs -f e2e_engine_0"
echo "  for p in \$(seq ${PORT_BASE} $((PORT_BASE+NUM_ENGINES-1))); do curl -s localhost:\$p/health && echo \" :\$p ok\"; done"
echo "全部 /health 200 后,启动 router: bash launch_router.sh"
