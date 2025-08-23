# vLLM Scheduler 性能建模

## 概述

本文档描述了用于预测 vLLM scheduler 中 model run 延迟的物理启发模型。该模型基于吞吐饱和理论，考虑了 batch_size 和 chunk_size（总 token 数）对计算延迟的影响。

## 问题背景

在 vLLM 的调度系统中，每个 batch 的 model run 延迟受多个因素影响：
- **Batch Size (B)**: 当前批次中的请求数量
- **Total Tokens (S)**: 当前批次需要计算的总 token 数（sum of chunk_sizes）
- **GPU 并行度**: SM 利用率、流水线填充、访存隐藏等
- **固定开销**: kernel launch、调度、拷贝等与规模相关的开销

## 模型理论

### 核心思想

我们将 GPU 计算建模为一个具有饱和特性的系统：
1. **有效吞吐量**随并行度增加而饱和
2. **工作量**主要由 token 数决定
3. **固定开销**随 batch 规模线性增长

### 数学表达式

```
Thr(B,S) = P_max * (1 - exp(-k_B * B)) * (1 - exp(-k_S * S))
Work(B,S) = w_0 + w_1 * S
T(B,S) = τ_0 + Work(B,S) / Thr(B,S) + τ_B * B + τ_S * S
```

其中：
- `T(B,S)`: 预测的 model run 延迟 (ms)
- `B`: batch_size（请求数量）
- `S`: 总 token 数（sum of chunk_sizes）

### 参数含义

#### 吞吐量参数
- **P_max**: 系统最大有效吞吐量（tokens/ms）
  - 物理含义：GPU 在理想并行条件下的峰值处理能力
  - 典型范围：1-1000

- **k_B**: batch 并行度敏感系数
  - 物理含义：batch_size 对并行度的影响强度
  - 值越大，达到饱和所需的 batch_size 越小
  - 典型范围：0.01-1.0

- **k_S**: token 并行度敏感系数
  - 物理含义：token 数对并行度的影响强度
  - 值越大，达到饱和所需的 token 数越小
  - 典型范围：0.001-0.1

#### 工作量参数
- **w_0**: 基础工作量常数
  - 物理含义：不依赖 token 数的固定计算开销
  - 典型范围：0-100

- **w_1**: 每 token 工作量系数
  - 物理含义：每个 token 的平均计算量
  - 典型范围：0.001-1.0

#### 线性开销参数
- **τ_0**: 基础延迟常数 (ms)
  - 物理含义：与规模无关的固定延迟（如 kernel launch）
  - 典型范围：0-50

- **τ_B**: 每 batch 额外延迟系数 (ms/batch)
  - 物理含义：每增加一个请求带来的额外开销
  - 典型范围：0-5

- **τ_S**: 每 token 额外延迟系数 (ms/token)
  - 物理含义：除计算外，每 token 的额外开销（如内存拷贝）
  - 典型范围：0-0.1

## 模型行为分析

### 边界行为

1. **小并行度 (B→0, S→0)**:
   - Thr ≈ P_max * k_B * B * k_S * S → 0
   - T → ∞（符合小批次效率低的观察）

2. **大并行度 (B→∞, S→∞)**:
   - Thr → P_max（饱和）
   - T ≈ τ_0 + (w_0 + w_1*S)/P_max + τ_B*B + τ_S*S
   - 延迟随规模线性增长（避免无限加速的不现实预测）

### 典型场景

1. **Decode-only**: S ≈ B，模型主要学习单个变量的影响
2. **Mixed workload**: S >> B，可以区分 batch 和 token 的不同效应
3. **Large prefill**: S >> B，主要由 token 数主导

## 拟合策略

### 数据预处理
1. **异常值过滤**: 移除延迟 > 300ms 或 < 1ms 的样本
2. **归一化**: 用中位数缩放 B 和 S，提升数值稳定性
3. **有效性检查**: 确保 B > 0, S > 0, T > 0

### 拟合方法
- **算法**: 非线性最小二乘法（scipy.optimize.curve_fit）
- **损失函数**: 平方误差（可选 Huber 损失以抗长尾噪声）
- **约束**: 所有参数 > 0（物理合理性）
- **初始化**: 基于数据分布的启发式估计

### 参数初始化策略
```python
P_max_init = percentile(S/T, 95)  # 近似峰值吞吐
k_B_init = 0.1                   # 中等敏感度
k_S_init = 0.01                  # 较低敏感度
w_0_init = 0.1                   # 小固定工作量
w_1_init = linear_fit(S, T).slope # 线性拟合斜率
τ_0_init = min(T) * 0.5          # 最小延迟的一半
τ_B_init = 1e-3                  # 小批次开销
τ_S_init = 1e-3                  # 小token开销
```

## 模型验证

### 拟合质量指标
- **R² 决定系数**: 解释方差比例
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **残差分析**: 检查系统性偏差

### 物理合理性检查
- 所有参数 > 0
- P_max 在合理范围内
- 饱和行为符合预期
- 边界条件合理

### 预测能力
- **内插**: 训练范围内的预测精度
- **外推**: 超出训练范围的预测可靠性
- **场景分析**: decode-only vs mixed workload

## 应用场景

### 性能优化
- 预测不同 batch_size 配置的延迟
- 找到最优的 batch/token 组合
- 识别性能瓶颈

### 容量规划
- 估算给定负载下的延迟分布
- SLA 保证分析
- 资源需求预测

### 调度策略
- 实时延迟预测
- 负载均衡决策
- 自适应批处理

## 使用示例

```python
from modeling.performance_model import ThroughputSaturationModel

# 创建模型
model = ThroughputSaturationModel()

# 拟合数据
model.fit(dataframe)

# 预测延迟
latency = model.predict(batch_size=16, total_tokens=512)

# 生成等高线图
model.plot_contour(batch_range=(1, 64), token_range=(1, 2048))
```

## 扩展方向

### 高阶效应
- 内存带宽瓶颈建模
- KV cache 命中率影响
- 异构硬件适配

### 多目标优化
- 延迟-吞吐量权衡
- 能耗建模
- 成本优化

### 动态建模
- 在线参数更新
- 自适应模型选择
- 异常检测 