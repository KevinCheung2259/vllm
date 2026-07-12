#!/bin/bash
# reasoning 行 lmetric 线:slo客户端 + maxtokens_file.txt + rounds1, qps50-58, 300s/点
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MRQA=/home/ubuntu/zhangy/mrqa
VENV=/home/ubuntu/zhangy/venv/bin/python3
OUTBASE="$SCRIPT_DIR/../../paper_figs/e2e_exp/data/reasoning/native_lmetric"
ts(){ date '+%F %T'; }

pid=$(ss -ltnp 2>/dev/null | grep ":8888 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$pid" ] && kill -9 "$pid"; sleep 2
( export PATH=/home/ubuntu/zhangy/venv/bin:$PATH
  setsid env PRODUCTION_STACK=/home/ubuntu/zhangy/production-stack MODEL=Qwen/QwQ-32B \
    NUM_ENGINES=8 ROUTING_LOGIC=lmetric VLLM_LMETRIC_EXACT_KV=false VLLM_LMETRIC_LIVE_BS=true \
    bash "$SCRIPT_DIR/launch_router.sh" </dev/null >/tmp/lmr3.out 2>&1 & disown )
for i in $(seq 1 20); do sleep 2
  [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8888/health 2>/dev/null)" = "200" ] && break
done
echo "[$(ts)] lmetric router up"

for Q in 50 52 54 56 58; do
  OUT="$OUTBASE/reasoning_native_lmetric_$Q"
  mkdir -p "$OUT"
  PQ=$(python3 -c "print($Q/5)")
  echo "[$(ts)] == qps $Q (5 x $PQ, full delivery) =="
  for i in 0 1 2 3 4; do
    timeout 400 "$VENV" "$MRQA/multi-round-qa-slo.py" \
      --dataset "$MRQA/reasoning.json" --maxtokens-file "$MRQA/maxtokens_file.txt" \
      --num-rounds 1 --answer-len 256 --qps "$PQ" --round-data 1000 \
      --init-user-id $((i*1000)) \
      --model Qwen/QwQ-32B --base-url http://localhost:8888/v1 \
      --api-key aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
      --output "$OUT/raw_$i.csv" --log-interval 60 --time 300 \
      > "$OUT/client_$i.log" 2>&1 &
    sleep 2
  done
  wait
  "$VENV" - "$OUT" <<'PY'
import sys, pandas as pd, os
d = sys.argv[1]
import glob as _g
fs = _g.glob(os.path.join(d, "raw*.csv"))
if fs:
    df = pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)
    df["launch_time"] = df["send_time"]
    df["finish_time"] = df["end_time"]
    df["generation_time"] = df["d_time"]
    df["generation_tokens"] = df["d_tokens"]
    df["prompt_tokens"] = df["p_tokens"]
    df.to_csv(os.path.join(d, "mapped.csv"), index=False)
    for x in fs: os.remove(x)
    print("mapped:", d, len(df))
PY
  echo "[$(ts)] qps $Q done"
  sleep 5
done
echo "[$(ts)] REASONING_LMETRIC_DONE"
