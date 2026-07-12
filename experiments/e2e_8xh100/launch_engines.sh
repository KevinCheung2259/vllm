#!/bin/bash
# =============================================================================
# 8×H100 引擎启动脚本 —— QwQ-32B,每卡 1 个实例,不使用 PP/TP
#
# 每个 vLLM 实例:
#   - 绑定单张 GPU (CUDA_VISIBLE_DEVICES=i),端口 8000+i
#   - 启用 SLA-aware 调度器 (VLLM_SLA_*)
#   - 启用 ELRAR Engine Agent,通过 UDP 单播把引擎状态推给 State Gateway
#
# 依赖:当前环境的 `vllm` 必须是本仓库 (exp-v0.9.1) 编译安装的版本,
#       否则没有 sla_aware / engine_agent。
#
# 用法:
#   bash launch_engines.sh          # 启动全部 8 个引擎(后台)
#   bash stop_all.sh                # 停止
# 日志:logs/engine_<i>.log,PID:logs/engine_<i>.pid
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ------------------------------- 可调配置 -----------------------------------
MODEL="${MODEL:-Qwen/QwQ-32B}"
SERVED_NAME="${SERVED_NAME:-Qwen/QwQ-32B}"   # 必须与 client / router 的 model 一致
NUM_ENGINES="${NUM_ENGINES:-8}"              # 引擎数 = GPU 数
PORT_BASE="${PORT_BASE:-8000}"               # 引擎端口从此起,engine i -> PORT_BASE+i
GPU_BASE="${GPU_BASE:-0}"                    # engine i -> GPU GPU_BASE+i
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# State Gateway 地址(单机部署时就是本机;Router 用 elrar 时会自动起 gateway)
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-9999}"

# SLA 目标(与 client 侧 SLO 保持一致)
SLO_TPOT_MS="${SLO_TPOT_MS:-50.0}"
SLO_TTFT_MS="${SLO_TTFT_MS:-1000.0}"
# 预拟合性能模型(可选,H100 6 参数模型已随 vllm 仓库提供)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/ubuntu/zhangy/vllm-workspace/vllm/vllm/v1/core/sched/sla_aware/fitted_model_h100_6param.pkl}"
# ---------------------------------------------------------------------------

echo "启动 ${NUM_ENGINES} 个 ${MODEL} 引擎 (GPU ${GPU_BASE}..$((GPU_BASE+NUM_ENGINES-1)), 端口 ${PORT_BASE}..$((PORT_BASE+NUM_ENGINES-1)))"
echo "State Gateway: ${GATEWAY_HOST}:${GATEWAY_PORT}"

for ((i=0; i<NUM_ENGINES; i++)); do
  gpu=$((GPU_BASE + i))
  port=$((PORT_BASE + i))
  engine_url="http://127.0.0.1:${port}"
  logf="${LOG_DIR}/engine_${i}.log"

  echo "  [engine ${i}] GPU=${gpu} port=${port} -> ${logf}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  VLLM_PORT="${port}" \
  \
  VLLM_SLA_SCHEDULER_ENABLED=true \
  VLLM_SLA_FALLBACK_ON_ERROR=true \
  VLLM_SLO_TPOT_MS="${SLO_TPOT_MS}" \
  VLLM_SLO_TTFT_MS="${SLO_TTFT_MS}" \
  VLLM_SLA_USE_PRETRAINED=true \
  VLLM_SLA_PRETRAINED_PATH="${PRETRAINED_MODEL}" \
  \
  VLLM_ENABLE_ELRAR=true \
  VLLM_ELRAR_NETWORK_MODE=unicast \
  VLLM_ELRAR_GATEWAY_HOST="${GATEWAY_HOST}" \
  VLLM_ELRAR_GATEWAY_PORT="${GATEWAY_PORT}" \
  VLLM_ELRAR_ENGINE_ID="${engine_url}" \
  VLLM_ELRAR_PUSH_INTERVAL=200 \
  \
  nohup vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${port}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-chunked-prefill \
    --disable-log-requests \
    > "${logf}" 2>&1 &

  echo $! > "${LOG_DIR}/engine_${i}.pid"
  sleep 2
done

echo ""
echo "全部引擎已后台启动。用以下命令观察就绪:"
echo "  tail -f ${LOG_DIR}/engine_0.log        # 看单个引擎日志"
echo "  for p in $((PORT_BASE))..$((PORT_BASE+NUM_ENGINES-1)); do curl -s localhost:\$p/health; done"
echo "等所有引擎 /health 返回 200 后,再启动 router: bash launch_router.sh"
