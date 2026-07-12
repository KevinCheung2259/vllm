#!/bin/bash
# =============================================================================
# 停止本实验启动的所有进程(引擎 / router / gateway),按 logs/*.pid 精确 kill。
# 用法:bash stop_all.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

kill_by_pidfile() {
  local pf="$1"
  [ -f "$pf" ] || return 0
  local pid; pid="$(cat "$pf" 2>/dev/null || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    echo "  停止 $(basename "$pf" .pid) (PID $pid)"
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$pf"
}

echo "停止 router / gateway ..."
kill_by_pidfile "${LOG_DIR}/router.pid"
kill_by_pidfile "${LOG_DIR}/gateway.pid"

echo "停止引擎(host 进程) ..."
for pf in "${LOG_DIR}"/engine_*.pid; do
  [ -e "$pf" ] || continue
  kill_by_pidfile "$pf"
done

echo "停止引擎(容器) ..."
for cf in "${LOG_DIR}"/engine_*.container; do
  [ -e "$cf" ] || continue
  name="$(cat "$cf" 2>/dev/null || true)"
  if [ -n "${name:-}" ]; then
    echo "  docker rm -f ${name}"
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
  rm -f "$cf"
done
# 兜底:按命名前缀清理
docker ps -a --filter "name=e2e_engine_" -q 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true

echo "完成。如仍有残留: pkill -f 'vllm_router.app'  /  docker ps --filter name=e2e_engine_"
