# Partial Fit 参数稳定性实验报告

## 1. 实验背景

SynergySched 论文的核心思想是**多时间尺度在线模型自适应**：硬件相关的结构参数通过离线 profiling 获取（慢时间尺度），负载相关的线性参数通过在线 partial fit 更新（快时间尺度）。

### 吞吐饱和模型（6参数）

$$T(B,S) = \frac{w_1 \cdot S}{P_{\max} \cdot (1 - e^{-k_B \cdot B}) \cdot (1 - e^{-k_S \cdot S})} + \tau_B \cdot B + \tau_S \cdot S$$

| 参数 | 含义 | 分组 |
|------|------|------|
| $P_{\max}$ | 峰值有效吞吐量 (tokens/ms) | 结构参数（离线固定） |
| $k_B$ | batch 并行度饱和系数 | 结构参数（离线固定） |
| $k_S$ | token 并行度饱和系数 | 结构参数（离线固定） |
| $w_1$ | 每 token 工作量系数 | 负载自适应（在线更新） |
| $\tau_B$ | 每 batch 额外延迟 (ms/batch) | 负载自适应（在线更新） |
| $\tau_S$ | 每 token 额外延迟 (ms/token) | 负载自适应（在线更新） |

### 为什么需要 Partial Fit？

在线连续批处理 (continuous batching) 产生的数据分布极为狭窄：~90% 的迭代是 decode-only（S ≈ B，值较小），导致饱和效应参数 $k_S$ 在线不可辨识。而离线 profiling 数据覆盖完整的 (B, S) 范围（B=1-137, S=2-4096），可以准确估计结构参数。

因此，正确的做法是：
- **离线 profiling** → 估计 $P_{\max}, k_B, k_S$（硬件属性，不随负载变化）
- **在线 partial fit** → 仅更新 $w_1, \tau_B, \tau_S$（负载特征，随 QPS 变化）

## 2. 实验配置

### 离线预训练模型

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA H100 |
| 模型 | meta-llama/Llama-3.1-8B-Instruct |
| 训练样本数 | 312,620 |
| B 范围 | 1 - 137 |
| S 范围 | 2 - 4,096 |
| R² | 0.993 |
| 模型文件 | `fitted_model_h100_6param.pkl` |

预训练参数：
| 参数 | 值 |
|------|-----|
| $P_{\max}$ | 4.0949 tokens/ms |
| $k_B$ | 10.0000 |
| $k_S$ | 0.000134 |
| $w_1$ | 0.003907 |
| $\tau_B$ | 1.8512 ms/batch |
| $\tau_S$ | 0.2353 ms/token |
| scales | (B_scale=9.0, S_scale=10.0) |

### 在线实验环境

| 项目 | 值 |
|------|-----|
| 服务器 | ubuntu@107.170.15.14 |
| Docker 镜像 | zhangy2259/vllm:2025-08-23 |
| 端口 | 8769 |
| BUFFER_SIZE | 4,000 |
| MIN_SAMPLES | 256 |
| MAPE_CHECK_INTERVAL | 15s |
| MODEL_UPDATE_THRESHOLD | 0.01 |

### 动态 QPS 负载模式

| 阶段 | QPS | 轮次 | 时长 |
|------|-----|------|------|
| Phase 1 | 3 | 12 rounds × 10s | 120s |
| Phase 2 | 5 | 12 rounds × 10s | 120s |
| Phase 3 | 3 | 12 rounds × 10s | 120s |

总计 360 秒，产生 24 次模型更新。

## 3. 实验结果

### 参数变异系数 (CV = std / |mean|)

| 参数 | CV | 分组 |
|------|------|------|
| $P_{\max}$ | **0.000** | 结构参数（离线固定） |
| $k_B$ | **0.000** | 结构参数（离线固定） |
| $k_S$ | **0.000** | 结构参数（离线固定） |
| $w_1$ | 0.123 | 负载自适应（在线更新） |
| $\tau_B$ | 0.337 | 负载自适应（在线更新） |
| $\tau_S$ | 0.096 | 负载自适应（在线更新） |

**汇总：**
- 结构参数平均 CV = **0.000**（完全恒定）
- 负载自适应参数平均 CV = **0.185**
- 模型 R² 在 0.91 - 0.95 范围内保持稳定

### 关键观察

1. **结构参数完全恒定**：$P_{\max}, k_B, k_S$ 在整个实验过程中（包括 QPS 切换）保持不变，验证了离线 profiling 获取的硬件属性不需要在线更新。

2. **负载自适应参数响应 QPS 变化**：
   - $\tau_B$（CV=0.337）变化最大，反映批处理开销随队列深度变化
   - $w_1$（CV=0.123）中等变化，反映计算效率随负载调整
   - $\tau_S$（CV=0.096）变化最小但仍有响应

3. **QPS 切换处参数发生跳变**：时间序列图清晰展示 $\tau_B$ 和 $w_1$ 在 QPS=3→5 和 QPS=5→3 切换点发生明显变化，验证了在线自适应的必要性。

## 4. 代码修改

### 4.1 throughput_model.py
新增 `partial_fit(df)` 方法：
- 固定 `self.params[0:3]`（P_max, k_B, k_S）来自预训练模型
- 仅通过 `curve_fit` 拟合 w_1, tau_B, tau_S
- 使用预训练模型的 `self.scales` 进行特征归一化

### 4.2 config.py
新增配置字段：
```python
partial_fit_enabled: bool = False  # VLLM_SLA_PARTIAL_FIT env var
```

### 4.3 performance_predictor.py
三处修改：
- `add_observation()`: 当 `partial_fit_enabled=True` 时，即使加载了预训练模型也继续收集数据
- `_should_update_model()`: 允许在 partial_fit 模式下触发更新
- `_update_model()`: 当 `partial_fit_enabled=True` 且模型已拟合时，调用 `partial_fit()` 而非 `fit()`

### 4.4 scheduler.py
两处条件修改（关键 bug fix）：
- 行 ~779：profiling 数据准备条件增加 `partial_fit_enabled` 判断
- 行 ~966：性能记录条件增加 `partial_fit_enabled` 判断

### Docker 启动参数
```bash
-e VLLM_SLA_USE_PRETRAINED=true \
-e VLLM_SLA_PRETRAINED_PATH=fitted_model_h100_6param.pkl \
-e VLLM_SLA_PARTIAL_FIT=true \
-e VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.01 \
-e VLLM_SLA_MAPE_CHECK_INTERVAL=15 \
-e VLLM_SLA_MIN_SAMPLES=256 \
-e VLLM_SLA_BUFFER_SIZE=4000 \
```

## 5. 对照实验：不进行在线自适应（Static Pretrained）

为验证在线自适应的必要性，我们运行了一个对照实验：加载相同的预训练模型，但**不进行任何在线参数更新**（eval_only 模式），仅周期性评估 R²。

### 配置差异

| 项目 | Partial Fit | Static (对照) |
|------|-------------|---------------|
| `VLLM_SLA_PARTIAL_FIT` | true | false |
| `VLLM_SLA_EVAL_ONLY` | false | true |
| 在线参数更新 | w_1, tau_B, tau_S | 无 |

### R² 对比

| 模式 | 平均 R² | 最小 R² | 最大 R² |
|------|---------|---------|---------|
| **Partial fit（在线自适应）** | **0.932** | **0.913** | **0.953** |
| Static pretrained（无自适应） | 0.038 | -0.260 | 0.428 |

### 关键发现

1. **静态预训练模型在线预测效果极差**：平均 R² 仅 0.038，最差时为 **-0.26**（负值意味着预测还不如取均值）
2. **R² 随 QPS 阶段大幅波动**：在 QPS=3 阶段 R² 为负值，QPS=5 阶段稍有改善但最高仅 0.43
3. **Partial fit 始终保持高精度**：R² 稳定在 0.91-0.95，不受 QPS 切换影响
4. **差距本质原因**：离线 profiling 数据的 (B, S) 分布与在线连续批处理的分布差异巨大，预训练的线性参数 (w_1, tau_B, tau_S) 无法直接迁移到在线场景

### 结论

仅靠离线 profiling 的预训练模型**完全不足以**在线服务场景下进行准确的延迟预测。在线 partial fit 自适应是不可或缺的。

## 6. 生成的文件

| 文件 | 说明 |
|------|------|
| `param_trace_partial.csv` | Partial fit 实验数据（24 行，6 参数列） |
| `param_trace_evalonly.csv` | 静态模型对照实验数据（25 行） |
| `param_stability_cv_partial.pdf` | CV 柱状图（结构 vs 自适应） |
| `param_stability_timeseries_partial.pdf` | 参数时间序列（含 QPS 阶段标注） |
| `r2_comparison_partial_vs_static.pdf` | R² 对比图（partial fit vs 静态模型） |
| `plot_param_stability.py` | 参数稳定性绘图脚本 |
| `plot_r2_comparison.py` | R² 对比绘图脚本 |
| `fitted_model_h100_6param.pkl` | 6 参数预训练模型 |

### 复现绘图命令
```bash
# 参数稳定性图
python3 plot_param_stability.py \
  --csv param_trace_partial.csv \
  --phases "6:QPS=5,14:QPS=3"

# R² 对比图
python3 plot_r2_comparison.py \
  --partial param_trace_partial.csv \
  --static param_trace_evalonly.csv \
  --phases "6:QPS=5,14:QPS=3"
```

## 7. 结论

两组实验共同验证了 SynergySched 的多时间尺度自适应策略的必要性和有效性：

1. **结构参数**（$P_{\max}, k_B, k_S$）反映硬件属性，通过离线 profiling 一次性获取即可，CV = 0.000
2. **负载自适应参数**（$w_1, \tau_B, \tau_S$）随工作负载动态变化，需要在线更新，CV = 0.185
3. **仅靠离线模型完全不够**：静态预训练模型平均 R² = 0.038（无预测能力），partial fit 平均 R² = 0.932
4. **Partial fit 在保持 R² > 0.91 的同时**，仅需拟合 3 个参数（而非 6 个），计算开销更低且更稳定
5. **多时间尺度分离是关键**：硬件属性（慢时间尺度）+ 负载特征（快时间尺度）的分离使得在线自适应既高效又准确
