#!/bin/bash
# 单卡 A/B:直连引擎 :8000(不经 router),隔离引擎层。
#   A(sc_lens):    SLA on + QwQ-32B stable 模型(我们的引擎)
#   B(sc_baseline):SLA off(基线引擎)
# 同一张卡、同 workload、同 QPS 点,对比引擎调度差异。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
WL="${WL:-sharegpt}"
QPS_LIST="${QPS_LIST:-1 2 3 4 6}"
DUR="${DUR:-300}"; PRELOAD="${PRELOAD:-15}"
PY=python3
ts(){ date '+%F %T'; }

run_config() {  # $1=config名 $2=SLA_ENABLED
  local cfg="$1" sla="$2"
  echo "########## [$(ts)] 配置 $cfg (SLA=$sla) ##########"
  docker rm -f e2e_engine_0 >/dev/null 2>&1 || true
  SLA_ENABLED="$sla" ELRAR_ENABLED=false NUM_ENGINES=1 MODEL=Qwen/QwQ-32B QUANT=fp8 \
    MAX_MODEL_LEN=16384 HF_CACHE=/mnt/local/hf-cache bash launch_engines_docker.sh >/dev/null 2>&1
  # 等就绪
  for i in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w %{http_code} localhost:8000/health 2>/dev/null)" = "200" ] && break
    sleep 10
  done
  echo "[$(ts)] $cfg 引擎就绪,开始扫点"
  for q in $QPS_LIST; do
    echo "[$(ts)]   $cfg QPS=$q ..."
    API_BASE="http://localhost:8000/v1" CONFIG="$cfg" QPS_LIST="$q" ROUND_DURATION="$DUR" MAX_ROUNDS=1 PRELOAD="$PRELOAD" \
      bash run_sweep.sh "$WL" >/dev/null 2>&1 || echo "[$(ts)]   QPS=$q 出错"
    csv="exp_dataset/${WL}_${cfg}/${WL}_qps${q}.csv"; n=0; [ -f "$csv" ] && n=$(($(wc -l < "$csv")-1))
    echo "[$(ts)]   $cfg QPS=$q -> 完成 $n (吞吐 $(awk -v n=$n -v d=$DUR 'BEGIN{printf "%.2f",n/d}') req/s)"
  done
}

run_config sc_lens true       # A: 我们的引擎(SLA on + QwQ 模型,launch 脚本默认已指 qwen32b)
run_config sc_baseline false  # B: 基线引擎(SLA off)
echo "[$(ts)] 单卡 A/B 全部完成。"
