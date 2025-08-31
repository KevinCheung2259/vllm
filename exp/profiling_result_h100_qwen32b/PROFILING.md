# 主要为了探究不同batch_size, chunk_size的情况下，一次迭代的时间

# vllm设置

export VLLM_SLA_SCHEDULER_ENABLED=false
export VLLM_ENABLE_ELRAR=false
export VLLM_ENABLE_SCHEDULER_PROFILING=true

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_chunk_256.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769  --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 200 --max-num-batched-tokens 256

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_chunk_512.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769  --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 200 --max-num-batched-tokens 512

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_chunk_1024.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769  --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 200 --max-num-batched-tokens 1024

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_chunk_2048.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769  --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 200 --max-num-batched-tokens 2048

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_chunk_4096.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769  --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 200 --max-num-batched-tokens 4096

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_chunk_8196.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769  --quantization fp8 --kv-cache-dtype fp8 --max-num-seqs 200 --max-num-batched-tokens 8196

# benchmark设置

cd zhangy/llm-inference-bench-char
python3 online_replay.py --input /home/ubuntu/replay-logs-origin.log --replay-mode qps --target-qps 1.5 --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model Qwen/QwQ-32B --max-token 200 --round-duration 60 --max-rounds 5

# exp

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_h100_qwen32b/scheduler_profiling_sched2.jsonl
vllm serve Qwen/QwQ-32B --max-model-len 10000 --disable-log-requests --port 8769 --ttft-slo 500 --tpot 50

cd zhangy/llm-inference-bench-char
python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode qps --target-qps 5 --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 60 --max-rounds 5 --e2e-slo 5 --ttft-slo 500 --tpot-slo 50

python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode timestamp --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 60 --max-rounds 5 --e2e-slo 5 --ttft-slo 500 --tpot-slo 50