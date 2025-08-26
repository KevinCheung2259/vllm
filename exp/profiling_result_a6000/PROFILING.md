# 主要为了探究不同batch_size, chunk_size, cache_token_num的情况下，一次迭代的时间

# vllm设置
export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a6000/scheduler_profiling_chunk_256.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 256 --tensor-parallel-size 2

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a6000/scheduler_profiling_chunk_512.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 512 --tensor-parallel-size 2

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a6000/scheduler_profiling_chunk_1024.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 1024 --tensor-parallel-size 2

export VLLM_SCHEDULER_PROFILING_LOG=vllm/exp/profiling_result_a6000/scheduler_profiling_chunk_2048.jsonl
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769 --max-num-seqs 200 --max-num-batched-tokens 2048 --tensor-parallel-size 2


# benchmark设置

cd llm-inference-benchmarking
python3 online_replay.py --input /mnt/shared/data/replay-logs-origin.log --replay-mode qps --target-qps 1.2 --sample-range 0 0.1  --api-base http://localhost:8769/v1 --model meta-llama/Llama-3.1-8B-Instruct --max-token 200 --round-duration 60 --max-rounds 5 


# 文件描述

scheduler_profiling_chunk_256_1.jsonl : target_qps=0.5,dataset=replay-logs-origin.log,time=5min
scheduler_profiling_chunk_256_2.jsonl : target_qps=1,dataset=replay-logs-origin.log,time=5min
scheduler_profiling_chunk_256_3.jsonl : target_qps=1.2,dataset=replay-logs-origin.log,time=5min

scheduler_profiling_chunk_512_1.jsonl : target_qps=1.2,dataset=replay-logs-origin.log,time=5min
scheduler_profiling_chunk_512_2.jsonl : target_qps=1.3,dataset=replay-logs-origin.log,time=5min

scheduler_profiling_chunk_1024_1.jsonl : target_qps=1.4,dataset=replay-logs-origin.log,time=5min
scheduler_profiling_chunk_1024_2.jsonl : target_qps=1.2,dataset=replay-logs-origin.log,time=5min

scheduler_profiling_chunk_2048_1.jsonl : target_qps=1.2,dataset=replay-logs-origin.log,time=5min
scheduler_profiling_chunk_2048_2.jsonl : target_qps=1.5,dataset=replay-logs-origin.log,time=5min
scheduler_profiling_chunk_2048_3.jsonl : target_qps=1.8,dataset=replay-logs-origin.log,time=5min
