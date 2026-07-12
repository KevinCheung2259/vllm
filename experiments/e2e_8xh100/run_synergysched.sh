#!/bin/bash
# SynergySched(SLA on + ELRAR)× 多数据集,自适应扫 QPS 到饱和。router(elrar)已在跑,不重启。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
export PATH="$HOME/zhangy/venv/bin:$PATH"
WORKLOADS="${WORKLOADS:-sharegpt code arxiv}"
CANDIDATE_QPS="${CANDIDATE_QPS:-4 8 12 16 20 24 32}"
DUR="${DUR:-300}"; PRELOAD="${PRELOAD:-20}"; SAT_RATIO="${SAT_RATIO:-1.08}"
ts(){ date '+%F %T'; }
echo "[$(ts)] SynergySched 扫描开始: workloads=[$WORKLOADS]"
for WL in $WORKLOADS; do
  JSONL="$SCRIPT_DIR/data/workloads/${WL}.jsonl"
  [ -f "$JSONL" ] || { echo "[$(ts)] 缺 $JSONL,跳过"; continue; }
  echo "############### [$(ts)] SynergySched / 数据集 $WL ###############"
  prev_thr=0
  for q in $CANDIDATE_QPS; do
    echo "[$(ts)]   $WL QPS=$q ..."
    CONFIG=synergysched QPS_LIST="$q" ROUND_DURATION="$DUR" MAX_ROUNDS=1 PRELOAD="$PRELOAD" \
      bash run_sweep.sh "$WL" >/dev/null 2>&1 || { echo "[$(ts)]   QPS=$q 出错,跳过"; continue; }
    csv="exp_dataset/${WL}_synergysched/${WL}_qps${q}.csv"
    n=0; [ -f "$csv" ] && n=$(($(wc -l < "$csv")-1))
    thr=$(awk -v n="$n" -v d="$DUR" 'BEGIN{printf "%.3f", n/d}')
    echo "[$(ts)]   $WL QPS=$q -> 完成 $n, 吞吐 ${thr} req/s"
    [ "$n" -eq 0 ] && { echo "[$(ts)]   完成 0,疑似故障,跳过 $WL"; break; }
    sat=$(awk -v c="$thr" -v p="$prev_thr" -v r="$SAT_RATIO" 'BEGIN{print (p>0 && c<p*r)?1:0}')
    [ "$sat" = "1" ] && { echo "[$(ts)]   $WL 已饱和(${thr} 未超 ${prev_thr}×${SAT_RATIO}),停止"; break; }
    prev_thr="$thr"
  done
done
echo "[$(ts)] SynergySched 全部完成。"
