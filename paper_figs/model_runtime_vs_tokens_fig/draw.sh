python draw_runtime_vs_tokens.py \
    "H100:/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result_h100" \
    "A100:/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result_a100" \
    "A6000-TP2:/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result_a6000" \
    "H100-32B:/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result_h100_qwen32b" \
    --rasterized --no-edges --point-size 30 --alpha 0.6 \
    --output multi_config_comparison.pdf