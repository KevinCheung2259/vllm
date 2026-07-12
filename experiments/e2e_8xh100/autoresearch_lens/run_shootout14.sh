#!/bin/bash
# =============================================================================
# qps=14 定点擂台:
#   phase1(引擎=老栈 SLA on,已在跑):
#     sla_elrar_v2   : router 在途修复(W5=1)
#     sla_elrar_v2t  : + 温和涡轮(DEEP_Q=30, TURBO=80;qps14 有富余容量,排完即关)
#   phase2(重启引擎 SLA off = sarathi):
#     native_rr      : roundrobin
#     native_session : session 路由(session-key=x-user-id,与老协议一致)
# 每点 300s,输出 /tmp/ar/shootout14/<cfg>
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # e2e_8xh100
KNOBS="$SCRIPT_DIR/autoresearch_lens/knobs.env"
OUT=/tmp/ar/shootout14
ts(){ date '+%F %T'; }

run_pt() {  # $1=cfg $2=router_logic $3=W5
  echo "[$(ts)] == $1 (router=$2 W5=$3) =="
  env ROUTING_LOGIC="$2" VLLM_ELRAR_W5="$3" VLLM_ELRAR_ADMIT_WINDOW=0 \
      CONFIG="$1" QPS_LIST="14" DUR=300 OUTROOT="$OUT/$1" \
      bash "$SCRIPT_DIR/run_fig6_flowgpt_q.sh"
}

mkdir -p "$OUT"

# phase 1: SLA-on engines
printf 'VLLM_SLA_DEEP_Q=0\n' > "$KNOBS"
run_pt sla_elrar_v2 elrar 1

printf 'VLLM_SLA_DEEP_Q=30\nVLLM_SLA_TURBO_MS=80\n' > "$KNOBS"
run_pt sla_elrar_v2t elrar 1
printf 'VLLM_SLA_DEEP_Q=0\n' > "$KNOBS"

# phase 2: restart engines SLA OFF (sarathi)
echo "[$(ts)] restarting engines SLA off ..."
docker rm -f $(docker ps -aq --filter name=e2e_engine) >/dev/null 2>&1
env NUM_ENGINES=8 MODEL=/root/.cache/huggingface/QwQ-32B SERVED_NAME=Qwen/QwQ-32B \
    QUANT=fp8 HF_CACHE=/mnt/local/models VLLM_SRC=/home/ubuntu/zhangy/vllm-workspace/vllm \
    SLA_ENABLED=false ELRAR_ENABLED=false MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.90 \
    bash "$SCRIPT_DIR/launch_engines_docker.sh" >/dev/null 2>&1
until [ "$(for p in $(seq 8000 8007); do curl -s -o /dev/null -w '%{http_code}' --max-time 3 localhost:$p/health 2>/dev/null; done | grep -o 200 | wc -l)" = 8 ]; do
  sleep 8
done
echo "[$(ts)] engines sarathi ready"

run_pt native_rr roundrobin 0
run_pt native_session session 0

echo "[$(ts)] SHOOTOUT14_DONE"
