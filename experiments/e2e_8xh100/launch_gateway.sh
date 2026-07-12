#!/bin/bash
# =============================================================================
# ELRAR State Gateway 独立启动脚本(可选)
#
# 注意:launch_router.sh 用 routing-logic=elrar 时会**在进程内自动启动**
#       gateway,通常不需要单独跑本脚本。仅在以下场景使用:
#         - 想让 gateway 独立进程/独立机器运行
#         - 调试:单独观察 UDP 状态聚合
#
# 用法:bash launch_gateway.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

PRODUCTION_STACK="${PRODUCTION_STACK:-/home/ubuntu/zhangy/production-stack}"
export VLLM_STATE_GATEWAY_UDP_PORT="${VLLM_STATE_GATEWAY_UDP_PORT:-9999}"
export VLLM_STATE_GATEWAY_STALE_THRESHOLD="${VLLM_STATE_GATEWAY_STALE_THRESHOLD:-2000}"
export VLLM_STATE_GATEWAY_NETWORK_MODE="${VLLM_STATE_GATEWAY_NETWORK_MODE:-unicast}"
export VLLM_STATE_GATEWAY_BIND_ADDRESS="${VLLM_STATE_GATEWAY_BIND_ADDRESS:-0.0.0.0}"

echo "启动 State Gateway: UDP :${VLLM_STATE_GATEWAY_UDP_PORT} (mode=${VLLM_STATE_GATEWAY_NETWORK_MODE})"
cd "${PRODUCTION_STACK}/src/vllm_router/services/state_gateway"
nohup python3 gateway.py > "${LOG_DIR}/gateway.log" 2>&1 &
echo $! > "${LOG_DIR}/gateway.pid"
echo "gateway 已后台启动 (PID $(cat "${LOG_DIR}/gateway.pid"))。日志:${LOG_DIR}/gateway.log"
