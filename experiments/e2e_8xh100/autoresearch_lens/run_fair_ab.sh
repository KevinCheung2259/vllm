#!/bin/bash
# =============================================================================
# 方案 A:同协议公平 A/B —— FlowGPT-Q qps14-18,480s/点,老栈引擎(涡轮关)。
#   线1 sla_elrar_fair:老栈 + 原版路由行为(VLLM_ELRAR_W5=0,关在途项)
#   线2 sla_elrar_v2  :老栈 + router 在途计数修复(W5=1)
# 输出到 paper_figs/e2e_exp/data/flowgpt_qps/<config>/,与 draw.py 兼容。
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # e2e_8xh100
DATA_ROOT="$SCRIPT_DIR/../../paper_figs/e2e_exp/data/flowgpt_qps"
ts(){ date '+%F %T'; }

run_line() {  # $1=config $2=W5
  local cfg="$1" w5="$2"
  echo "[$(ts)] ===== line $cfg (W5=$w5) ====="
  rm -rf "$DATA_ROOT/$cfg"
  env ROUTING_LOGIC=elrar VLLM_ELRAR_W5="$w5" VLLM_ELRAR_ADMIT_WINDOW=0 \
      CONFIG="$cfg" QPS_LIST="14 15 16 17 18" DUR=300 \
      OUTROOT="$DATA_ROOT/$cfg" \
      bash "$SCRIPT_DIR/run_fig6_flowgpt_q.sh"
  echo "[$(ts)] ===== line $cfg done ====="
}

run_line sla_elrar_fair 0
run_line sla_elrar_v2   1

echo "[$(ts)] AB_ALL_DONE"
