#!/bin/bash
# =============================================================================
# v4 完整 Fig6 线 + RR 基线线(同协议 300s/点,引擎=sarathi+ELRAR agent,已在跑):
#   1) synergy_v4  FlowGPT-Q qps15-18(qps14 从擂台复制)
#   2) synergy_v4  FlowGPT-T 0.28-0.36
#   3) native_rr_fair FlowGPT-Q qps15-18(qps14 从擂台复制)
#   4) native_rr_fair FlowGPT-T 0.28-0.36
# 数据落 paper_figs/e2e_exp/data/{flowgpt_qps,flowgpt_timestamp}/<cfg>/
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DQ="$SCRIPT_DIR/../../paper_figs/e2e_exp/data/flowgpt_qps"
DT="$SCRIPT_DIR/../../paper_figs/e2e_exp/data/flowgpt_timestamp"
ts(){ date '+%F %T'; }

# 复用擂台 qps14 数据
mkdir -p "$DQ/synergy_v4" "$DQ/native_rr_fair"
cp -r /tmp/ar/shootout14/synergy_v4/flowgpt_qps_synergy_v4_14 "$DQ/synergy_v4/" 2>/dev/null || true
mkdir -p "$DQ/native_rr_fair/flowgpt_qps_native_rr_fair_14"
cp /tmp/ar/shootout14/native_rr/flowgpt_qps_native_rr_14/* "$DQ/native_rr_fair/flowgpt_qps_native_rr_fair_14/" 2>/dev/null || true

echo "[$(ts)] === v4 FlowGPT-Q 15-18 ==="
env ROUTING_LOGIC=elrar VLLM_ELRAR_RR_VETO=1 VLLM_ELRAR_VETO_MARGIN=2 VLLM_ELRAR_ADMIT_WINDOW=0 \
    CONFIG=synergy_v4 QPS_LIST="15 16 17 18" DUR=300 OUTROOT="$DQ/synergy_v4" \
    bash "$SCRIPT_DIR/run_fig6_flowgpt_q.sh"

echo "[$(ts)] === v4 FlowGPT-T 0.28-0.36 ==="
env ROUTING_LOGIC=elrar VLLM_ELRAR_RR_VETO=1 VLLM_ELRAR_VETO_MARGIN=2 VLLM_ELRAR_ADMIT_WINDOW=0 \
    CONFIG=synergy_v4 UPPERS="0.28 0.30 0.32 0.34 0.36" DUR=300 OUTROOT="$DT/synergy_v4" \
    bash "$SCRIPT_DIR/run_fig6_flowgpt_t.sh"

echo "[$(ts)] === rr FlowGPT-Q 15-18 ==="
env ROUTING_LOGIC=roundrobin \
    CONFIG=native_rr_fair QPS_LIST="15 16 17 18" DUR=300 OUTROOT="$DQ/native_rr_fair" \
    bash "$SCRIPT_DIR/run_fig6_flowgpt_q.sh"

echo "[$(ts)] === rr FlowGPT-T 0.28-0.36 ==="
env ROUTING_LOGIC=roundrobin \
    CONFIG=native_rr_fair UPPERS="0.28 0.30 0.32 0.34 0.36" DUR=300 OUTROOT="$DT/native_rr_fair" \
    bash "$SCRIPT_DIR/run_fig6_flowgpt_t.sh"

echo "[$(ts)] V4_FULL_DONE"
