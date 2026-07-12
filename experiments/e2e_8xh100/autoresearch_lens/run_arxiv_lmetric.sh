#!/bin/bash
# arxiv 行 lmetric 线:slo客户端 ans180 rounds1 拆分x3,qps12-16,300s/点,bf16引擎
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MRQA=/home/ubuntu/zhangy/mrqa
VENV=/home/ubuntu/zhangy/venv/bin/python3
OUTBASE="$SCRIPT_DIR/../../paper_figs/e2e_exp/data/arxiv/native_lmetric"
ts(){ date '+%F %T'; }

# lmetric router
pid=$(ss -ltnp 2>/dev/null | grep ":8888 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$pid" ] && kill -9 "$pid"; sleep 2
( export PATH=/home/ubuntu/zhangy/venv/bin:$PATH
  setsid env PRODUCTION_STACK=/home/ubuntu/zhangy/production-stack MODEL=Qwen/QwQ-32B \
    NUM_ENGINES=8 ROUTING_LOGIC=lmetric VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true \
    bash "$SCRIPT_DIR/launch_router.sh" </dev/null >/tmp/lmr2.out 2>&1 & disown )
for i in $(seq 1 20); do sleep 2
  [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)" = "200" ] && break
done
echo "[$(ts)] lmetric router up"

for Q in 12 13 14 15 16; do
  OUT="$OUTBASE/arxiv_native_lmetric_$Q"
  mkdir -p "$OUT"
  PQ=$(python3 -c "print($Q/3)")
  echo "[$(ts)] == qps $Q (3 x $PQ) =="
  for i in 0 1 2; do
    timeout 400 "$VENV" "$MRQA/multi-round-qa-slo.py" \
      --dataset "$MRQA/arxiv_5k_to_9k.json" --num-rounds 1 --answer-len 180 \
      --qps "$PQ" --round-data 1000 --init-user-id $((i*1000)) \
      --model Qwen/QwQ-32B --base-url http://localhost:8888/v1 \
      --api-key aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
      --output "$OUT/raw_$i.csv" --log-interval 60 --time 300 \
      > "$OUT/client_$i.log" 2>&1 &
    sleep 3
  done
  wait
  "$VENV" - "$OUT" <<'PY'
import sys, pandas as pd, os, glob
d = sys.argv[1]
fs = glob.glob(os.path.join(d, "raw_*.csv"))
if fs:
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["launch_time"] = df["send_time"]
    df["finish_time"] = df["end_time"]
    df["generation_time"] = df["d_time"]
    df["generation_tokens"] = df["d_tokens"]
    df["prompt_tokens"] = df["p_tokens"]
    df.to_csv(os.path.join(d, "mapped.csv"), index=False)
    for f in fs: os.remove(f)
    print("mapped:", d, len(df))
PY
  echo "[$(ts)] qps $Q done"
  sleep 5
done
echo "[$(ts)] ARXIV_LMETRIC_DONE"
