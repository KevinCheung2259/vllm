# Adaptive Model Ablation Experiment

消融实验：证明多时间尺度（Adaptive）在线更新优于单一时间尺度（Long-window / Short-window）。

## 服务器信息

- **服务器**: ubuntu@107.170.15.14
- **GPU**: H100
- **vLLM 镜像**: zhangy2259/vllm:2025-08-23
- **模型**: meta-llama/Llama-3.1-8B-Instruct
- **代码目录**: /home/ubuntu/zhangy/vllm-workspace/vllm/
- **Benchmark 脚本**: /home/ubuntu/zhangy/llm-inference-benchmarking/online_replay.py
- **Replay 数据**: /mnt/shared/data/replay-logs-origin.log

## 文件说明

| 文件 | 说明 |
|------|------|
| adaptive-qps5.json | Adaptive 实验结果（NDJSON，每行一轮） |
| offline-qps5.json | Offline 预训练模型实验结果 |
| native-qps5.json | Native vLLM（无 SLA 调度器）实验结果 |
| long_window-qps5.json | Long-window Only 消融实验结果 |
| short_window-qps5.json | Short-window Only 消融实验结果 |
| plot_latency_p50_json.py | 绘图脚本：读取上述 JSON 生成 p50 latency 对比图 |
| plot_latency_comparison.py | 参考绘图脚本（基于 CSV 的旧版） |

## 代码改动

消融实验需要在 SLA 调度器中新增 mape_check_interval 配置项：

### 1. config.py

**文件**: vllm/v1/core/sched/sla_aware/config.py

- 新增字段: mape_check_interval: float = 30.0
- from_env() 中读取环境变量 VLLM_SLA_MAPE_CHECK_INTERVAL，默认 30

### 2. performance_predictor.py

**文件**: vllm/v1/core/sched/sla_aware/performance_predictor.py

- __init__() 中将硬编码 self.mape_check_interval = 30.0 改为 self.mape_check_interval = config.mape_check_interval

## 实验参数

### 共同参数（所有实验一致）

Benchmark 参数:
    --replay-mode qps --target-qps 5 --sample-range 0 0.1
    --max-token 200 --round-duration 10 --max-rounds 12
    --e2e-slo 2.5 --ttft-slo 500 --tpot-slo 50

共同 Docker 环境变量:
    VLLM_SLA_SCHEDULER_ENABLED=true
    VLLM_SLA_USE_PRETRAINED=false
    VLLM_SLA_USE_STABLE_MODEL=false
    VLLM_SLA_VERBOSE=true
    VLLM_SLA_FALLBACK_ON_ERROR=true
    VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.08

### 各实验差异参数

| 参数 | Adaptive | Long-window Only | Short-window Only |
|------|----------|-----------------|-------------------|
| VLLM_SLA_MIN_SAMPLES | 256 | 2048 | 32 |
| VLLM_SLA_BUFFER_SIZE | 1024 | 2048 | 32 |
| VLLM_SLA_MAPE_CHECK_INTERVAL | 15 | 60 | 2 |
| VLLM_SLA_MODEL_CONFIDENCE | 0.9 | 0.9 | 0.5 |

### 参数设计思路

- **Adaptive**: MIN_SAMPLES=256 使模型在约 50s 时有足够样本开始训练；BUFFER_SIZE=1024 保证拟合质量；MAPE_CHECK_INTERVAL=15s 适度频率检查模型质量
- **Long-window Only**: MIN_SAMPLES=2048 在 90s 内无法积累足够样本（QPS=5 约产生约 450 条），模型始终无法更新，一直用 fallback 线性模型
- **Short-window Only**: MIN_SAMPLES=32 + BUFFER_SIZE=32 极小窗口导致拟合不稳定；MAPE_CHECK_INTERVAL=2s 频繁重训；MODEL_CONFIDENCE=0.5 接受劣质模型用于调度

### 对照组

- **Offline**: 使用预训练模型 VLLM_SLA_USE_PRETRAINED=true，无在线更新
- **Native**: 原生 vLLM 调度器，VLLM_SLA_SCHEDULER_ENABLED=false

## 运行实验

### Step 1: 停止当前容器

    docker stop zy_docker

### Step 2: 启动实验容器（以 Adaptive 为例）

    docker run -d --rm       --name zy_adaptive       --gpus all       -v /home/ubuntu/zhangy/vllm-workspace:/vllm-workspace       -p 8769:8769       -e VLLM_SLA_SCHEDULER_ENABLED=true       -e VLLM_SLA_USE_PRETRAINED=false       -e VLLM_SLA_USE_STABLE_MODEL=false       -e VLLM_SLA_VERBOSE=true       -e VLLM_SLA_FALLBACK_ON_ERROR=true       -e VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.08       -e VLLM_SLA_MODEL_CONFIDENCE=0.9       -e VLLM_SLA_MIN_SAMPLES=256       -e VLLM_SLA_BUFFER_SIZE=1024       -e VLLM_SLA_MAPE_CHECK_INTERVAL=15       -e VLLM_ELRAR_NETWORK_MODE=unicast       -e VLLM_ELRAR_ENGINE_ID=http://65.49.81.73:8769       -e VLLM_ENABLE_ELRAR=true       -e VLLM_ELRAR_GATEWAY_HOST=184.105.190.123       -e VLLM_ELRAR_PUSH_INTERVAL=100       -e VLLM_ELRAR_GATEWAY_PORT=9999       zhangy2259/vllm:2025-08-23       -c "vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 10000 --disable-log-requests --port 8769"

替换对应环境变量即可运行 Long-window / Short-window 实验。

### Step 3: 等待服务就绪（约 60s）

    docker logs --tail 3 zy_adaptive
    # 看到 "Application startup complete." 即可

### Step 4: 运行 benchmark

**注意**: benchmark 脚本以追加模式写入 JSON，运行前先删除旧文件。

    rm -f /home/ubuntu/zhangy/vllm-workspace/vllm/paper_figs/adaptive_model_exp/adaptive-qps5.json

    python3 /home/ubuntu/zhangy/llm-inference-benchmarking/online_replay.py       --input /mnt/shared/data/replay-logs-origin.log       --replay-mode qps --target-qps 5 --sample-range 0 0.1       --api-base http://localhost:8769/v1       --model meta-llama/Llama-3.1-8B-Instruct       --max-token 200 --round-duration 10 --max-rounds 12       --e2e-slo 2.5 --ttft-slo 500 --tpot-slo 50       --json-output /home/ubuntu/zhangy/vllm-workspace/vllm/paper_figs/adaptive_model_exp/adaptive-qps5.json

### Step 5: 停止实验容器，恢复原始容器

    docker stop zy_adaptive
    docker start zy_docker

### Step 6: 生成图表

    cd /home/ubuntu/zhangy/vllm-workspace/vllm/paper_figs/adaptive_model_exp
    python3 plot_latency_p50_json.py
    # 输出: latency_p50_qps5.pdf

## 实验结果（2026-02-07）

| 方法 | 稳定期均值 (30-90s) | 相对 Adaptive |
|------|-------------------|--------------|
| Offline | 3.43s | -6.5% |
| **Adaptive** | **3.67s** | **基准** |
| Long-window Only | 3.71s | +1.1% |
| Short-window Only | 4.08s | +11.2% |
| Native | 4.72s | +28.6% |

### 结论

单一时间尺度（无论长窗口还是短窗口）都不如多时间尺度 Adaptive 方案：
- **Long-window**: 窗口过大导致 90s 内无法完成模型更新，始终依赖 fallback 线性模型
- **Short-window**: 窗口过小 + 频繁更新导致模型拟合不稳定，调度决策噪声大
