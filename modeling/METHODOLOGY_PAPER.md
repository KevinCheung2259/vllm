# GPU推理系统吞吐饱和建模理论（论文方法部分）

## 1. 研究动机与问题定义

在大规模生成式模型的推理系统中（如 vLLM），单步推理延迟受批内请求数与本步需计算的 token 总量双重驱动。我们以批大小 B（batch size）与总 token 数 S（sum of chunk sizes）为控制变量，刻画单步推理延迟 T 的可解释、可拟合、可外推的物理启发模型，旨在为性能分析、SLA 反解与调度优化提供统一的理论基础。

**记号定义**：
- $B \in \mathbb{N}^+$: 本步批内请求数
- $S \in \mathbb{N}^+$: 本步需要计算的总 token 数（跨请求求和）
- $T(B,S)$: 单步 model run 的时间（ms）

**目标**：构建物理可解释且统计稳健的函数 $T(B,S)$，并给出参数识别与估计方法。

## 2. 吞吐饱和理论（Throughput Saturation）

GPU 的并行计算呈现典型的"随并行度增长而趋于饱和"的特征：在低并行度下，新增负载显著提升单位时间处理能力（吞吐）；当并行度足够高后，硬件算力、内存带宽与流水线调度等成为瓶颈，使得吞吐量逼近平台上限。该现象可用指数趋近形式刻画。

我们将"有效吞吐量"表示为 $\mathrm{Thr}(B,S)$，并将并行度的两个显著来源（批内并行 B 与 token 并行 S）以乘性饱和相结合：

$$\mathrm{Thr}(B,S) = P_{\max}\bigl(1-e^{-k_B B}\bigr)\bigl(1-e^{-k_S S}\bigr)$$

其中 $P_{\max}>0$ 为平台峰值吞吐（tokens/ms），$k_B,k_S>0$ 控制从低并行到饱和的上升速率。该形式反映了：
- **B 增大**可提高 SM 占用与 warp 并发
- **S 增大**可改善算子批量化与访存隐藏
- **两者的综合作用**受硬件与带宽上限抑制，呈现联合饱和

## 3. 工作量与延迟分解

### 3.1 工作量建模

延迟可分解为"可并行的计算工作量/有效吞吐量"与"不可并行或近似线性的开销"之和。构造工作量对 S 近似线性（固定隐藏维与模型规模下的 FLOPs 与序列长度成比例）：

$$\mathrm{Work}(B,S) = w_0 + w_1 S, \quad w_0 \geq 0, \; w_1 > 0$$

### 3.2 总延迟模型

$$T(B,S) = \tau_0 + \frac{\mathrm{Work}(B,S)}{\mathrm{Thr}(B,S)} + \tau_B B + \tau_S S$$

其中：
- $\tau_0 \geq 0$ 表示与负载无关的固定开销（kernel 启动、runtime/driver 固定成本等）
- $\tau_B \geq 0$ 捕捉随 B 增长的近线性额外开销（调度/同步/内存元数据处理等）
- $\tau_S \geq 0$ 捕捉随 S 增长的近线性额外开销（数据搬运、缓存 miss 诱发的访存开销、非理想流水化等）

### 3.3 三项式结构的理论依据

该三项式结构与经典性能分析的一致性：
- **服务时间** ≈ 工作量/服务速率（吞吐）
- **固定与线性修正项**补足"不可并行或难以饱和"的系统开销
- **物理上限**由 $P_{\max}$ 主导，增长速率由 $k_B,k_S$ 控制

## 4. 边界与单调性性质

### 4.1 边界行为分析

**低并行度极限**：当 $B,S \to 0$ 时，$\mathrm{Thr} \to 0$，故 $T \to +\infty$：低并行下效率极低

**高并行度极限**：当 $B,S \to +\infty$ 时，$\mathrm{Thr} \to P_{\max}$，延迟近似
$$T \approx \tau_0 + \frac{w_0+w_1 S}{P_{\max}} + \tau_B B + \tau_S S$$

呈现线性增长，避免不现实的"无限并行导致零延迟"。

### 4.2 单调性保证

此外，$T$ 对 B、S 分别单调不增/不增的"单位工作量平均用时"体现为：吞吐随并行增长而边际收益递减，与经验一致。

## 5. 可辨识性与可解释性

### 5.1 参数物理意义

- $P_{\max}$ 反映平台吞吐上限
- $k_B,k_S$ 反映达到饱和所需并行规模
- $w_1$ 对应"每 token 平均计算量"的量级
- $w_0$ 表示不随 S 变的算子/编排成本
- $\tau_0,\tau_B,\tau_S$ 分别对应固定开销、批内管理开销与 token 线性开销

### 5.2 可辨识性策略

为提高可辨识性，建议采集覆盖 decode-only（$S \approx B$）、混合 prefill/decode 与大 prefill 的多样数据，以打破变量共线。同时可对参数设定正约束与合理先验。

## 6. 参数估计方法

### 6.1 统计模型

将观测的 $(B_i,S_i,T_i)$ 视为
$$T_i = T(B_i,S_i;\theta) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,\sigma^2)$$

其中 $\theta=\{P_{\max},k_B,k_S,w_0,w_1,\tau_0,\tau_B,\tau_S\}$。

### 6.2 优化求解

采用带边界的非线性最小二乘（如 trust-region reflective）求解：

**预处理步骤**：
1. 去除明显异常（如 $T>300$ms 的长尾）
2. 保证 $B,S,T>0$
3. 以中位数缩放 $B,S$ 以改善数值条件

**初值设定**：
- $P_{\max}$ 取 $\mathrm{quantile}(S/T,0.95)$
- $k_B,k_S \in [0.01,0.5]$
- $w_1$ 由对 $S \to T$ 的线性回归斜率初始化
- $\tau_0 \approx \min(T)/2$
- $\tau_B,\tau_S$ 取小正数

**约束条件**：
- 对所有参数施加非负下界，防止非物理解

**评价指标**：
- 报告 $R^2$、RMSE、MAE
- 做残差诊断与交叉验证

### 6.3 鲁棒性增强

若需鲁棒性，可采用 Huber/Tukey 损失；若误差异方差显著，可对目标施加权重（如对不同 $S$ 段加权）。

## 7. 与替代模型的比较

### 7.1 现有方法的局限

- **线性/多项式回归**：缺乏物理上限与饱和行为，外推差、易过拟合、可解释性弱
- **简单吞吐常数模型**：无法反映并行度随 B、S 增长而变化

### 7.2 本方法优势

本模型以"Work/Thr + 线性修正 + 常数"的最小充分结构，既嵌入硬件饱和机理，又保留必要的系统开销通道，兼顾可解释性、拟合性与外推性。

## 8. 误差分解与敏感性

### 8.1 误差来源分析

可将误差分解为：
- **模型结构偏差**：吞吐饱和项或线性项不足以覆盖的高阶效应（如缓存级联、bank 冲突）
- **参数不确定性**：有限样本与共线性导致的方差
- **度量噪声**：计时粒度、异步开销抖动

### 8.2 敏感性分析

建议进行：
- **灵敏度分析**：对 $B,S$ 网格评估 $\partial T/\partial B$、$\partial T/\partial S$
- **参数置信区间**：由协方差矩阵近似或 bootstrap 获得
- **分区评估**：在 decode-only/混合/大 prefill 区间分别评估拟合度

## 9. 实践扩展（可选）

### 9.1 高阶修正

若观测到在极大 $B,S$ 下的非单调或"反直觉变慢"，可引入轻量校正项：
$$T(B,S) = \tau_0 + \frac{w_0+w_1 S}{\mathrm{Thr}(B,S)}\bigl(1+\gamma_B B+\gamma_S S\bigr) + \tau_B B + \tau_S S$$

其中 $\gamma_B,\gamma_S \geq 0$ 捕捉高并行导致的带宽/互连等干扰，但建议在验证基础模型不足时再启用。

### 9.2 参数初始化算法

```
算法：参数初始化策略
输入：观测数据 {(B_i, S_i, T_i)}_{i=1}^N
输出：初始参数 θ_0

1. 数据预处理：
   - 过滤异常值：T ∈ [1ms, 300ms]
   - 归一化：B_norm = B/median(B), S_norm = S/median(S)

2. 启发式估计：
   - P_max_init ← percentile(S/T, 95)
   - 线性拟合：w_1_init ← slope(regress(S → T))
   - τ_0_init ← min(T) × 0.5
   - k_B_init, k_S_init ← 0.1, 0.01
   - τ_B_init, τ_S_init ← 1e-3
   - w_0_init ← w_1_init × median(S) × 0.1

3. 约束检查：
   - 确保所有参数 > 0
   - 确保 P_max 在合理范围 [1, 10000]

返回 θ_0 = {P_max_init, k_B_init, k_S_init, w_0_init, w_1_init, τ_0_init, τ_B_init, τ_S_init}
```

## 10. 典型应用

### 10.1 配置外推与容量规划

用拟合后的 $T(B,S)$ 预测未见配置的延迟表现。

### 10.2 SLA 反解

固定目标时延 $T^\star$，给定 $B$ 反解 $S$（或反之），用于在线调度决策：

$$S^\star = \arg\min_S |T(B, S) - T^\star|$$

### 10.3 策略评估

对不同批策略（decode 优先、chunked prefill 策略、waiting 启动门槛）做等高线与前沿分析。

### 10.4 实时调度示例

```python
def schedule_batch_with_sla(model, current_requests, sla_target):
    """
    基于模型预测进行SLA约束的批调度
    """
    batch_size = len(current_requests)
    total_tokens = sum(req.remaining_tokens for req in current_requests)
    
    # 预测当前配置延迟
    predicted_latency = model.predict(batch_size, total_tokens)
    
    if predicted_latency <= sla_target:
        return current_requests  # 满足SLA
    else:
        # 减少token数或批大小以满足SLA
        return optimize_for_sla(model, current_requests, sla_target)
```

## 11. 模型验证框架

### 11.1 交叉验证策略

```
算法：时间序列交叉验证
输入：时间序列数据 {(B_t, S_t, T_t)}_{t=1}^T
参数：min_train_size, test_window_size

1. For t = min_train_size to T-test_window_size:
   a. 训练集：{(B_i, S_i, T_i)}_{i=1}^t
   b. 测试集：{(B_i, S_i, T_i)}_{i=t+1}^{t+test_window_size}
   c. 拟合模型 θ_t ← fit(训练集)
   d. 评估：error_t ← evaluate(θ_t, 测试集)

2. 报告平均性能：mean(error_t)
```

### 11.2 残差分析

```python
def residual_analysis(model, data):
    """
    模型残差分析
    """
    predictions = model.predict(data['B'], data['S'])
    residuals = data['T'] - predictions
    
    # 1. 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # 2. 异方差检验
    bp_stat, bp_p = stats.breusch_pagan(residuals, data[['B', 'S']])
    
    # 3. 自相关检验
    dw_stat = durbin_watson(residuals)
    
    # 4. 线性趋势检验
    trend_slope, trend_p = stats.linregress(range(len(residuals)), residuals)[:2]
    
    return {
        'normality': {'statistic': shapiro_stat, 'p_value': shapiro_p},
        'heteroscedasticity': {'statistic': bp_stat, 'p_value': bp_p},
        'autocorrelation': {'durbin_watson': dw_stat},
        'trend': {'slope': trend_slope, 'p_value': trend_p}
    }
```

## 12. 结论

本文提出的吞吐饱和驱动的延迟模型

$$T(B,S) = \tau_0 + \frac{w_0+w_1 S}{P_{\max}(1-e^{-k_B B})(1-e^{-k_S S})} + \tau_B B + \tau_S S$$

以最简洁的形式统一了"并行度带来的吞吐提升（至饱和）"与"系统性固定/线性开销"，参数具备明确物理意义，拟合稳健且外推性良好。实验显示，该模型可在多种负载形态（decode-only、混合、长 prefill）下取得高 $R^2$、低 RMSE，并有效支撑 SLA 反解与调度优化。

### 主要贡献总结

1. **理论创新**：首次将硬件吞吐饱和特性引入 LLM 推理延迟建模
2. **模型优势**：兼具物理可解释性、统计拟合性与工程实用性
3. **应用价值**：为性能预测、容量规划、实时调度提供统一理论基础

---

**备注**：本方法论文档可直接纳入学术论文的方法部分。如需完整实现代码、详细实验结果或扩展应用，请参考配套的技术文档与代码库。 