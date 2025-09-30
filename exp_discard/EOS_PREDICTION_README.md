# EOS Token概率预测实验

这个实验旨在探究是否能够根据每次迭代后EOS token的概率来预测request何时结束。

## 功能概述

我们修改了vLLM的v1调度器，添加了记录每次生成步骤中EOS token概率的功能。这些数据可以用于：

1. 分析EOS概率的变化模式
2. 研究EOS概率与请求结束的关系
3. 建立预测模型来提前预知请求何时会结束

## 修改的文件

### 1. `vllm/v1/core/sched/scheduler.py`

主要修改：
- 添加了EOS概率记录的配置选项
- 在`update_from_output`方法中增加了`_log_eos_probability`调用
- 实现了`_log_eos_probability`方法来提取和记录EOS token概率

关键新增代码：
```python
# EOS token概率记录设置
self.enable_eos_prob_logging = os.getenv('VLLM_ENABLE_EOS_PROB_LOGGING', 'true').lower() == 'true'
self.eos_prob_log_file = os.getenv('VLLM_EOS_PROB_LOG_FILE', 'eos_probabilities.jsonl')

def _log_eos_probability(self, request: Request, new_logprobs, token_index: int) -> None:
    """记录当前token的EOS概率"""
    # 提取EOS token概率并记录到文件
```

## 环境变量配置

使用以下环境变量来控制EOS概率记录功能：

```bash
# 启用EOS概率记录（默认启用）
export VLLM_ENABLE_EOS_PROB_LOGGING=true

# 设置日志文件路径（默认为eos_probabilities.jsonl）
export VLLM_EOS_PROB_LOG_FILE=vllm/predict_exp/eos_probabilities.jsonl
```

## 记录的数据格式

每条记录是一个JSON对象，包含以下字段：

```json
{
  "request_id": "请求唯一标识符",
  "timestamp": 1234567890.123,
  "step": 5,
  "current_token_id": 123,
  "eos_token_id": 2,
  "eos_logprob": -2.3026,
  "eos_prob": 0.1000,
  "is_finished": false,
  "finish_reason": null,
  "prompt_length": 20,
  "output_length": 5,
  "total_length": 25
}
```

字段说明：
- `request_id`: 请求的唯一标识符
- `timestamp`: 记录时间戳
- `step`: 当前生成步骤（输出token数量）
- `current_token_id`: 当前生成的token ID
- `eos_token_id`: EOS token的ID
- `eos_logprob`: EOS token的对数概率（如果在top-k中）
- `eos_prob`: EOS token的概率（2^eos_logprob）
- `is_finished`: 请求是否已完成
- `finish_reason`: 完成原因（如果已完成）
- `prompt_length`: 提示长度
- `output_length`: 当前输出长度
- `total_length`: 总长度

## 使用方法

### 1. 运行推理测试

使用提供的测试脚本：

```bash
cd /home/paperspace/zhangy/vllm-workspace/vllm
python test_eos_prediction.py
```

这个脚本会：
- 设置环境变量
- 运行一系列测试提示
- 记录EOS概率数据
- 进行基本分析

### 2. 分析EOS概率数据

使用分析脚本来深入分析记录的数据：

```bash
python analyze_eos_prediction.py
```

这个脚本会：
- 加载记录的EOS概率数据
- 分析概率变化趋势
- 生成可视化图表
- 计算预测特征的区分能力

### 3. 自定义使用

在您自己的推理代码中：

```python
import os
from vllm import LLM, SamplingParams

# 启用EOS概率记录
os.environ['VLLM_ENABLE_EOS_PROB_LOGGING'] = 'true'
os.environ['VLLM_EOS_PROB_LOG_FILE'] = 'my_eos_data.jsonl'

# 创建模型（确保启用logprobs）
llm = LLM(model="your-model-name")
sampling_params = SamplingParams(
    max_tokens=100,
    logprobs=10,  # 重要：必须启用logprobs才能记录EOS概率
    temperature=0.8
)

# 运行推理
outputs = llm.generate(["你的提示"], sampling_params)

# EOS概率会自动记录到指定文件
```

## 分析和预测

### 可视化分析

分析脚本会生成以下图表：

1. **典型请求的EOS概率变化**：显示不同请求在生成过程中EOS概率的变化
2. **最终EOS概率分布**：所有请求最终EOS概率的直方图
3. **EOS概率趋势 vs 序列长度**：EOS概率变化趋势与序列长度的关系
4. **平均EOS概率 vs 最大EOS概率**：不同请求的EOS概率统计特征

### 预测特征

以下特征可用于预测请求结束：

1. **平均EOS概率**：前几步的平均EOS概率
2. **最大EOS概率**：前几步的最大EOS概率
3. **EOS概率标准差**：EOS概率的变化程度
4. **当前EOS概率**：最新的EOS概率
5. **序列长度**：当前生成的token数量

### 预测模型建议

基于记录的数据，您可以：

1. **统计模型**：使用阈值或简单规则
   ```python
   if eos_prob > 0.5 or avg_eos_prob > 0.3:
       prediction = "即将结束"
   ```

2. **机器学习模型**：
   - 逻辑回归
   - 随机森林
   - 神经网络

3. **时序模型**：
   - LSTM/GRU网络
   - Transformer模型

## 注意事项

1. **logprobs必须启用**：只有当`sampling_params.logprobs > 0`时才能记录EOS概率
2. **top-k限制**：EOS概率只有在EOS token出现在top-k tokens中时才能记录
3. **性能影响**：记录功能会略微增加推理延迟和存储开销
4. **文件权限**：确保程序有权限写入指定的日志文件

## 实验思路

### 1. 数据收集
- 在不同类型的任务上收集EOS概率数据
- 收集不同长度、不同复杂度的请求数据

### 2. 模式发现
- 分析EOS概率在不同类型请求中的变化模式
- 识别预示请求即将结束的信号

### 3. 模型训练
- 使用历史数据训练预测模型
- 评估模型的准确率和预测提前量

### 4. 应用优化
- 根据预测结果优化资源调度
- 提前准备下一个请求的资源
- 动态调整批处理策略

## 扩展可能性

1. **实时预测**：集成预测模型到调度器中
2. **多模态特征**：结合其他token的概率信息
3. **上下文感知**：考虑提示内容对EOS概率的影响
4. **动态调整**：根据预测结果动态调整采样参数

## 故障排除

### 常见问题

1. **没有EOS概率记录**
   - 检查是否启用了logprobs
   - 确认EOS token在top-k范围内

2. **日志文件为空**
   - 检查文件权限
   - 确认环境变量设置正确

3. **分析脚本报错**
   - 安装必要的依赖：`pip install matplotlib numpy pandas`
   - 检查数据文件格式

### 调试技巧

1. 启用详细日志：
   ```python
   import logging
   logging.getLogger('vllm').setLevel(logging.DEBUG)
   ```

2. 检查环境变量：
   ```python
   import os
   print("EOS logging:", os.getenv('VLLM_ENABLE_EOS_PROB_LOGGING'))
   print("Log file:", os.getenv('VLLM_EOS_PROB_LOG_FILE'))
   ```

## 联系信息

如果您在使用过程中遇到问题或有改进建议，请通过以下方式联系：

- 提交Issue到项目仓库
- 发送邮件描述问题和使用场景

希望这个实验能帮助您更好地理解和预测语言模型的生成行为！ 