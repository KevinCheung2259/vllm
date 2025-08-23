# 主要为了探究不同batch_size, chunk_size的情况下，一次迭代的时间

# vllm设置

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a100/scheduler_profiling_chunk_256.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 256

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a100/scheduler_profiling_chunk_512.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 512

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a100/scheduler_profiling_chunk_1024.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 1024

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a100/scheduler_profiling_chunk_2048.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 2048

# benchmark设置

cd zhangy/llm-inference-bench-char
python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode qps --target-qps 1.5 --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 60 --max-rounds 5 --e2e-slo 2.5

# exp

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a100/scheduler_profiling_sched2.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --ttft-slo 500 --tpot 50

cd zhangy/llm-inference-bench-char
python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode qps --target-qps 5 --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 60 --max-rounds 5 --e2e-slo 5 --ttft-slo 500 --tpot-slo 50

python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode timestamp --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 60 --max-rounds 5 --e2e-slo 5 --ttft-slo 500 --tpot-slo 50