# Adaptive Model Experiment

## Run vllm

```bash

#!/usr/bin/env bash
# run_vllm.sh 
# 按需把 ${port} 改成具体端口，或者通过命令行 ./run_vllm.sh 8769 传入

set -euo pipefail

# 1. 如果没有通过命令行传参，则默认 8769
PORT=${1:-8769}

# 2. 构造容器名（可选）
CONTAINER_NAME="zy_docker"

# 3. 启动容器并执行 vllm serve
docker run --gpus all -d \
  --name "${CONTAINER_NAME}" \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v /home/paperspace/zhangy/vllm-workspace:/vllm-workspace \
  -e VLLM_SLA_SCHEDULER_ENABLED=true \
  -e VLLM_SLA_FALLBACK_ON_ERROR=true \
  -e VLLM_SLA_USE_STABLE_MODEL=false \
  -e VLLM_SLA_USE_PRETRAINED=false \
  -e VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.08 \
  -e VLLM_SLA_MIN_SAMPLES=1024 \
  -e VLLM_SLA_BUFFER_SIZE=1024 \
  -e VLLM_SLA_MODEL_CONFIDENCE=0.9 \
  -e VLLM_SLA_VERBOSE=true \
  -e VLLM_ENABLE_ELRAR=true \
  -e VLLM_ELRAR_NETWORK_MODE=unicast \
  -e VLLM_ELRAR_GATEWAY_HOST=184.105.190.123 \
  -e VLLM_ELRAR_GATEWAY_PORT=9999 \
  -e VLLM_ELRAR_PUSH_INTERVAL=100 \
  -e VLLM_ELRAR_ENGINE_ID="http://65.49.81.73:${PORT}" \
  --entrypoint /bin/bash \
  zhangy2259/vllm:2025-08-23 \
  -c "vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port ${PORT}"

```

## Run benchmark

```bash
python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode qps --target-qps 5 --sample-range 0 0.1 --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 10 --max-rounds 30 --e2e-slo 2.5 --ttft-slo 500 --tpot-slo 50 --detailed-logs adaptive_model_exp/native-qps.csv --json-output adaptive_model_exp/native-qps.json
```
## 