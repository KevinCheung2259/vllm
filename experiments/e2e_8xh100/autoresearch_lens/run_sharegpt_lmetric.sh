#!/bin/bash
# =============================================================================
# sharegpt 行 lmetric 正式线(破解的协议):
#   引擎: QwQ-32B FP8 + fp8 KV cache, sarathi(SLA off)
#   客户端: multi-round-qa-slo.py, rounds=2, answer-len=256, round-data=1000
#   路由: lmetric;qps 40/44/48/52 连跑各 300s
#   输出: CSV 列映射为 csv_process 格式后落 paper_figs/.../sharegpt/native_lmetric/
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MRQA=/home/ubuntu/zhangy/mrqa
VENV=/home/ubuntu/zhangy/venv/bin/python3
OUTBASE="$SCRIPT_DIR/../../paper_figs/e2e_exp/data/sharegpt/native_lmetric"
ts(){ date '+%F %T'; }

# lmetric router
pid=$(ss -ltnp 2>/dev/null | grep ":8888 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$pid" ] && kill -9 "$pid"; sleep 2
( export PATH=/home/ubuntu/zhangy/venv/bin:$PATH
  setsid env PRODUCTION_STACK=/home/ubuntu/zhangy/production-stack MODEL=Qwen/QwQ-32B \
    NUM_ENGINES=8 ROUTING_LOGIC=lmetric VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true \
    bash "$SCRIPT_DIR/launch_router.sh" </dev/null >/tmp/lmr.out 2>&1 & disown )
for i in $(seq 1 20); do sleep 2
  [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)" = "200" ] && break
done
echo "[$(ts)] lmetric router up"

for Q in 40 44 48 52; do
  OUT="$OUTBASE/sharegpt_native_lmetric_$Q"
  mkdir -p "$OUT"
  echo "[$(ts)] == qps $Q =="
  timeout 400 "$VENV" "$MRQA/multi-round-qa-slo.py" \
    --dataset "$MRQA/sharegpt.json" --num-rounds 2 --answer-len 256 --qps "$Q" \
    --round-data 1000 --model Qwen/QwQ-32B --base-url http://localhost:8888/v1 \
    --api-key aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
    --output "$OUT/raw.csv" --log-interval 30 --time 300 \
    > "$OUT/client.log" 2>&1 || true
  # 列映射 -> csv_process 非-flowgpt 格式
  "$VENV" - "$OUT" <<'PY'
import sys, pandas as pd, os
d = sys.argv[1]
f = os.path.join(d, "raw.csv")
if os.path.exists(f):
    df = pd.read_csv(f)
    df["launch_time"] = df["send_time"]
    df["finish_time"] = df["end_time"]
    df["generation_time"] = df["d_time"]
    df["generation_tokens"] = df["d_tokens"]
    df["prompt_tokens"] = df["p_tokens"]
    df.to_csv(os.path.join(d, "mapped.csv"), index=False)
    os.remove(f)
    print("mapped:", d, len(df))
PY
  echo "[$(ts)] qps $Q done"
  sleep 5
done
echo "[$(ts)] SHAREGPT_LMETRIC_DONE"
