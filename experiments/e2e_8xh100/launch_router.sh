#!/bin/bash
# =============================================================================
# production-stack vllm_router 启动脚本 —— ELRAR 智能路由
#
# 用 ELRAR 路由时,router 会在进程内**自动启动 State Gateway** (UDP :9999),
# 接收各引擎 EngineAgent 推来的实时状态并据此打分选路。
# 客户端 (online_replay.py) 打 router 的 :8888。
#
# 前置:先跑 launch_engines.sh,等 8 个引擎 /health OK。
#
# 依赖:production-stack 的 exp 分支(含 ELRARRouter + state_gateway)。
# 用法:
#   bash launch_router.sh                 # routing-logic=elrar(默认)
#   ROUTING_LOGIC=qps bash launch_router.sh   # 换其它策略做对比
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ------------------------------- 可调配置 -----------------------------------
PRODUCTION_STACK="${PRODUCTION_STACK:-/home/ubuntu/zhangy/production-stack}"
MODEL="${MODEL:-Qwen/QwQ-32B}"               # 必须与引擎 served-model-name 一致
NUM_ENGINES="${NUM_ENGINES:-8}"
PORT_BASE="${PORT_BASE:-8000}"
ROUTER_PORT="${ROUTER_PORT:-8888}"
ROUTING_LOGIC="${ROUTING_LOGIC:-elrar}"      # elrar|qps|roundrobin|session|least_loaded|latency_based|weight_based
GATEWAY_PORT="${GATEWAY_PORT:-9999}"         # ELRAR 路由自动起的 gateway UDP 端口

# ELRAR 打分权重与参数(可调;默认见 ELRARRouter)
export VLLM_ELRAR_SLO_MS="${VLLM_ELRAR_SLO_MS:-50}"
export VLLM_ELRAR_STALE_MS="${VLLM_ELRAR_STALE_MS:-3000}"
export VLLM_ELRAR_W1="${VLLM_ELRAR_W1:-1.0}"   # latency
export VLLM_ELRAR_W2="${VLLM_ELRAR_W2:-1.0}"   # load
export VLLM_ELRAR_W3="${VLLM_ELRAR_W3:-1.0}"   # mode match
export VLLM_ELRAR_W4="${VLLM_ELRAR_W4:-1.0}"   # kv affinity
export VLLM_STATE_GATEWAY_UDP_PORT="${GATEWAY_PORT}"
# ---------------------------------------------------------------------------

# 拼出 8 个后端 URL:http://localhost:8000,...,http://localhost:8007
backends=""
models=""
mtypes=""
for ((i=0; i<NUM_ENGINES; i++)); do
  port=$((PORT_BASE + i))
  backends+="http://localhost:${port},"
  models+="${MODEL},"
  mtypes+="chat,"
done
backends="${backends%,}"; models="${models%,}"; mtypes="${mtypes%,}"

echo "启动 vllm_router: port=${ROUTER_PORT} routing-logic=${ROUTING_LOGIC}"
echo "后端: ${backends}"
[ "${ROUTING_LOGIC}" = "elrar" ] && echo "ELRAR 模式:将自动启动 State Gateway (UDP :${GATEWAY_PORT})"

# session 类策略需要 session-key
EXTRA_ARGS=()
if [ "${ROUTING_LOGIC}" = "session" ] || [ "${ROUTING_LOGIC}" = "cache_aware_load_balancing" ] || [ "${ROUTING_LOGIC}" = "elrar" ]; then
  # ELRAR 也需要 session-key(用于 KV 亲和打分;不传会因 headers.get(None) 崩溃)
  EXTRA_ARGS+=(--session-key "${SESSION_KEY:-x-user-id}")
fi
if [ "${ROUTING_LOGIC}" = "latency_based" ]; then
  EXTRA_ARGS+=(--latency-type "${LATENCY_TYPE:-e2e}")
fi
if [ "${ROUTING_LOGIC}" = "weight_based" ] && [ -n "${ENGINE_WEIGHTS:-}" ]; then
  EXTRA_ARGS+=(--engine-weights "${ENGINE_WEIGHTS}")
fi

cd "${PRODUCTION_STACK}/src"
PYTHONPATH="${PRODUCTION_STACK}/src:${PYTHONPATH:-}" \
nohup python3 -m vllm_router.app \
  --port "${ROUTER_PORT}" \
  --service-discovery static \
  --static-backends "${backends}" \
  --static-models "${models}" \
  --static-model-types "${mtypes}" \
  --routing-logic "${ROUTING_LOGIC}" \
  "${EXTRA_ARGS[@]}" \
  --engine-stats-interval 10 \
  --request-stats-window 10 \
  --log-stats \
  > "${LOG_DIR}/router.log" 2>&1 &

echo $! > "${LOG_DIR}/router.pid"
echo "router 已后台启动 (PID $(cat "${LOG_DIR}/router.pid"))。日志:${LOG_DIR}/router.log"
echo "就绪后即可跑客户端: bash run_all_clients.sh"
