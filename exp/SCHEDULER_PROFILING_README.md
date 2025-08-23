# vLLM Scheduler Profiling 功能

这个功能可以帮助您详细分析vLLM调度器的性能，记录每个批次的调度信息，包括prefill数量、decode数量、chunk size和调度时间等。

## 功能特性

- 📊 **详细统计**: 记录每批次的prefill/decode请求数量
- ⏱️ **性能分析**: 测量调度器执行时间和模型运行时间
- 📦 **Chunk Size追踪**: 分析prefill和decode的token chunk大小
- 📈 **趋势分析**: 支持时间序列分析和可视化
- 🔧 **灵活配置**: 通过环境变量轻松启用/禁用
- ⚡ **完整时间链路**: 从调度到模型执行的完整时间分析

## 记录的信息

每个调度批次会记录以下信息:

```json
{
  "batch_id": 123,                    // 批次ID
  "timestamp": 1703123456.789,        // 时间戳
  "schedule_duration_ms": 5.2,        // 调度耗时(毫秒)
  "model_run_duration_ms": 45.8,      // Model Run耗时(毫秒)
  "total_step_duration_ms": 51.0,     // 总Step耗时(毫秒)
  "num_prefill_reqs": 3,              // Prefill请求数
  "num_decode_reqs": 7,               // Decode请求数  
  "total_scheduled_tokens": 2048,     // 总调度token数
  "prefill_chunk_sizes": [512, 256, 128], // Prefill chunk大小列表
  "decode_chunk_sizes": [1, 1, 1, 1, 1, 1, 1], // Decode chunk大小列表
  "avg_prefill_chunk_size": 298.67,   // 平均prefill chunk大小
  "max_prefill_chunk_size": 512,      // 最大prefill chunk大小
  "min_prefill_chunk_size": 128,      // 最小prefill chunk大小
  "avg_decode_chunk_size": 1.0,       // 平均decode chunk大小
  "num_waiting_reqs": 5,              // 等待队列中的请求数
  "num_running_reqs": 10,             // 运行中的请求数
  "kv_cache_usage": 0.75              // KV cache使用率
}
```

## 使用方法

### 1. 启用Profiling

通过环境变量启用profiling功能:

```bash
export VLLM_ENABLE_SCHEDULER_PROFILING=true
export VLLM_SCHEDULER_PROFILING_LOG=scheduler_profiling.jsonl
export VLLM_SCHEDULER_PROFILING_CONSOLE=true
```

### 2. 运行vLLM服务器

```bash
python -m vllm.entrypoints.openai.api_server \
    --model your_model_name \
    --host 0.0.0.0 \
    --port 8000
```

### 3. 发送测试请求

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_name",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### 4. 分析Profiling数据

使用提供的分析脚本:

```bash
python scheduler_profiling_example.py analyze scheduler_profiling.jsonl
```

## 环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_ENABLE_SCHEDULER_PROFILING` | `false` | 是否启用profiling |
| `VLLM_SCHEDULER_PROFILING_LOG` | `scheduler_profiling.jsonl` | 日志文件路径 |
| `VLLM_SCHEDULER_PROFILING_CONSOLE` | `false` | 是否在控制台输出统计信息 |

## 分析工具

提供的`scheduler_profiling_example.py`脚本包含:

### 基本统计
- 总批次数
- 平均/最大/最小调度时间
- 平均prefill/decode请求数
- 平均chunk size分析

### 可视化图表
- 调度时间vs模型运行时间趋势对比图
- Prefill vs Decode请求数分布
- 总Token数趋势
- 调度时间分布直方图
- 模型运行时间分布直方图
- 调度时间vs模型运行时间相关性散点图

## 实际应用场景

### 1. 性能优化
通过分析调度时间趋势，识别性能瓶颈:
```bash
# 查看调度时间超过10ms的批次
grep '"schedule_duration_ms":[0-9][0-9]\.' scheduler_profiling.jsonl
```

### 2. Chunk Size调优
分析不同chunk size对性能的影响:
- 观察prefill chunk size分布
- 对比不同配置下的throughput

### 3. 容量规划  
通过请求数量趋势预测资源需求:
- 监控waiting queue长度
- 分析KV cache使用率

### 4. 异常检测
识别调度异常情况:
- 调度时间突然增长
- 请求积压在waiting queue

## 示例分析

```python
import json
import pandas as pd

# 读取profiling数据
data = []
with open('scheduler_profiling.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# 分析调度效率
print(f"平均调度时间: {df['schedule_duration_ms'].mean():.2f}ms")
print(f"P95调度时间: {df['schedule_duration_ms'].quantile(0.95):.2f}ms")

# 分析模型运行效率
if 'model_run_duration_ms' in df.columns:
    print(f"平均模型运行时间: {df['model_run_duration_ms'].mean():.2f}ms")
    print(f"P95模型运行时间: {df['model_run_duration_ms'].quantile(0.95):.2f}ms")
    print(f"调度占总时间比例: {(df['schedule_duration_ms'].mean() / df['total_step_duration_ms'].mean()):.2%}")

# 分析throughput
total_time_ms = df['total_step_duration_ms'].sum() if 'total_step_duration_ms' in df.columns else df['schedule_duration_ms'].sum()
print(f"平均tokens/秒: {df['total_scheduled_tokens'].sum() / total_time_ms * 1000:.2f}")

# 分析负载特征
print(f"Prefill比例: {df['num_prefill_reqs'].sum() / (df['num_prefill_reqs'].sum() + df['num_decode_reqs'].sum()):.2%}")
```

## 注意事项

1. **性能影响**: Profiling会增加少量开销，建议在测试环境使用
2. **磁盘空间**: 长时间运行会产生大量日志，注意磁盘空间
3. **数据格式**: 使用JSONL格式，每行一个JSON对象，便于流式处理
4. **时区**: 时间戳使用Unix时间戳，注意时区转换

## 故障排除

### 日志文件为空
- 检查环境变量设置
- 确认vLLM进程有写文件权限
- 检查是否有实际调度发生

### 分析脚本报错
- 安装依赖: `pip install pandas matplotlib`
- 检查日志文件格式是否正确
- 确认Python版本兼容性

### 控制台无输出
- 检查`VLLM_SCHEDULER_PROFILING_CONSOLE`环境变量
- 查看vLLM日志级别设置 