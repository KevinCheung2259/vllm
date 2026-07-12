#!/bin/bash
# =============================================================================
# 基线矩阵编排(无人值守):引擎保持 SLA OFF,遍历 {数据集 × router 方法},
# 每个组合自适应扫 QPS 直到饱和。结果按 <workload>_<router>_baseline 分目录存。
#
# 前置:8 引擎已用 SLA_ENABLED=false ELRAR_ENABLED=false 起好(本脚本不动引擎)。
# 用法: nohup bash run_baseline_matrix.sh > ~/zhangy/matrix.log 2>&1 &
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/zhangy/venv/bin:$PATH"

WORKLOADS="${WORKLOADS:-code arxiv}"
ROUTERS="${ROUTERS:-roundrobin least_loaded latency_based}"
CANDIDATE_QPS="${CANDIDATE_QPS:-4 8 12 16 20 24 32 48 64}"
DUR="${DUR:-300}"
PRELOAD="${PRELOAD:-20}"
SAT_RATIO="${SAT_RATIO:-1.08}"
PS="${PRODUCTION_STACK:-$HOME/zhangy/production-stack}"

ts(){ date '+%F %T'; }
echo "[$(ts)] 矩阵开始: workloads=[$WORKLOADS] routers=[$ROUTERS] 候选QPS=[$CANDIDATE_QPS]"

for WL in $WORKLOADS; do
  JSONL="$SCRIPT_DIR/data/workloads/${WL}.jsonl"
  [ -f "$JSONL" ] || { echo "[$(ts)] 缺 $JSONL,跳过数据集 $WL"; continue; }
  echo "############### [$(ts)] 数据集 $WL ###############"
  for router in $ROUTERS; do
    echo "========== [$(ts)] $WL / ROUTER=$router =========="
    kill "$(cat logs/router.pid 2>/dev/null)" 2>/dev/null || true
    sleep 3
    ROUTING_LOGIC="$router" PRODUCTION_STACK="$PS" bash launch_router.sh >/dev/null 2>&1
    sleep 12
    for _ in $(seq 1 10); do
      [ "$(curl -s -o /dev/null -w %{http_code} localhost:8888/health 2>/dev/null)" = "200" ] && break
      sleep 3
    done
    prev_thr=0
    for q in $CANDIDATE_QPS; do
      echo "[$(ts)]   $WL/$router QPS=$q ..."
      CONFIG="${router}_baseline" QPS_LIST="$q" ROUND_DURATION="$DUR" MAX_ROUNDS=1 PRELOAD="$PRELOAD" \
        bash run_sweep.sh "$WL" >/dev/null 2>&1 || { echo "[$(ts)]   QPS=$q 出错,跳过"; continue; }
      csv="exp_dataset/${WL}_${router}_baseline/${WL}_qps${q}.csv"
      n=0; [ -f "$csv" ] && n=$(($(wc -l < "$csv")-1))
      thr=$(awk -v n="$n" -v d="$DUR" 'BEGIN{printf "%.3f", n/d}')
      echo "[$(ts)]   $WL/$router QPS=$q -> 完成 $n, 吞吐 ${thr} req/s"
      # 完成数为 0 视为异常(router/引擎问题),跳出该组合避免空转
      [ "$n" -eq 0 ] && { echo "[$(ts)]   $WL/$router QPS=$q 完成 0,疑似故障,跳过该组合"; break; }
      saturated=$(awk -v c="$thr" -v p="$prev_thr" -v r="$SAT_RATIO" 'BEGIN{print (p>0 && c < p*r)?1:0}')
      if [ "$saturated" = "1" ]; then
        echo "[$(ts)]   $WL/$router 已饱和(${thr} 未超 ${prev_thr}×${SAT_RATIO}),停止"
        break
      fi
      prev_thr="$thr"
    done
  done
done
echo "[$(ts)] 基线矩阵全部完成。"
