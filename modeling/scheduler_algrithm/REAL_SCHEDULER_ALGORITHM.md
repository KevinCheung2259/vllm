# vLLM 真实调度场景的反向优化算法

## 🎯 问题重新定义

基于scheduler.py的实际调度逻辑，重新定义反向优化问题。

### 真实输入结构

#### 1. Running Queue (`self.running`)
正在运行的请求，包含两类：
- **Decode阶段请求**: `num_computed_tokens >= num_prompt_tokens`
  - 每步只需要1个token（生成下一个token）
  - 优先级最高，必须优先满足
- **Chunked Prefill阶段请求**: `num_computed_tokens < num_prompt_tokens`  
  - 需要继续处理剩余的prompt tokens
  - 受`long_prefill_threshold`限制单次chunk大小

#### 2. Waiting Queue (`self.waiting`)
等待队列中的新请求：
- 都是`num_computed_tokens = 0`的新请求
- 需要开始prefill阶段
- 按优先级和到达顺序调度

### 约束条件

1. **Token预算约束**: `total_tokens ≤ max_num_scheduled_tokens`
2. **并发限制**: `len(running) + len(new_scheduled) ≤ max_num_running_reqs`
3. **阶段约束**: 
   - Decode请求: `chunk_size = 1`
   - Prefill请求: `1 ≤ chunk_size ≤ min(remaining_tokens, long_prefill_threshold)`
4. **优先级约束**: Running > Waiting，Decode > Prefill

### 输出结果
- **Running请求调度**: `Dict[req_id, chunk_size]`
- **Waiting请求选择**: `List[selected_req_ids]` 
- **Waiting请求分配**: `Dict[req_id, chunk_size]`
- **性能预测**: 预测延迟、可行性等

## 🧮 算法设计

### 核心思想

**三阶段优化策略**：
1. **阶段1**: 估算最优batch_size和total_tokens
2. **阶段2**: 优先调度running队列（decode > prefill）
3. **阶段3**: 贪心选择和调度waiting队列

### 详细算法流程

#### 阶段1: 全局优化
```python
def estimate_optimal_config(target_latency):
    best_config = None
    best_error = inf
    
    # 遍历可能的batch_size
    for batch_size in range(current_running, max_running + 1):
        # 数值求解最优token数
        optimal_tokens = solve_equation(
            latency_function(batch_size, S) = target_latency
        )
        
        # 评估该配置的可行性和误差
        config = simulate_scheduling(batch_size, optimal_tokens)
        error = abs(config.predicted_latency - target_latency)
        
        if error < best_error:
            best_config = config
            best_error = error
    
    return best_config
```

#### 阶段2: Running队列调度（最高优先级）
```python
def schedule_running_requests(running_reqs, token_budget):
    chunk_sizes = {}
    remaining_tokens = token_budget
    
    # 1. 优先调度decode请求（固定1 token）
    decode_reqs = [req for req in running_reqs if req.is_decode_phase]
    for req in decode_reqs:
        if remaining_tokens >= 1:
            chunk_sizes[req.request_id] = 1
            remaining_tokens -= 1
        else:
            chunk_sizes[req.request_id] = 0
    
    # 2. 调度chunked prefill请求
    prefill_reqs = [req for req in running_reqs if not req.is_decode_phase]
    prefill_reqs.sort(key=lambda r: (-r.priority, -r.remaining_tokens))
    
    for req in prefill_reqs:
        max_chunk = min(
            req.remaining_prompt_tokens,
            long_prefill_threshold,
            remaining_tokens
        )
        chunk_sizes[req.request_id] = max_chunk
        remaining_tokens -= max_chunk
    
    return chunk_sizes, remaining_tokens
```

#### 阶段3: Waiting队列选择（贪心策略）
```python
def select_waiting_requests(waiting_reqs, available_tokens, available_slots):
    # 按优先级排序
    sorted_waiting = sorted(waiting_reqs, key=lambda r: -r.priority)
    
    selected = []
    chunk_sizes = {}
    remaining_tokens = available_tokens
    
    for req in sorted_waiting:
        if len(selected) >= available_slots or remaining_tokens <= 0:
            break
        
        # 计算最小启动成本
        min_chunk = min(64, req.num_prompt_tokens)  # 最小chunk
        if remaining_tokens < min_chunk:
            break
        
        # 贪心分配
        allocation = min(
            req.num_prompt_tokens,
            long_prefill_threshold, 
            remaining_tokens
        )
        
        selected.append(req.request_id)
        chunk_sizes[req.request_id] = allocation
        remaining_tokens -= allocation
    
    return selected, chunk_sizes
```

## 🔧 算法特点

### 1. 符合真实调度逻辑
- **队列结构**: 严格按照running/waiting分离
- **优先级策略**: decode > chunked_prefill > new_prefill
- **约束处理**: 完全符合vLLM的实际限制

### 2. 高效的搜索策略
- **有界搜索**: batch_size搜索空间有限
- **快速求解**: 每个batch_size下的token数可快速求解
- **早期终止**: 找到满足精度要求的解即可停止

### 3. 实际可部署
- **接口兼容**: 输出格式直接对应scheduler需要的决策
- **参数可控**: 支持各种调度策略参数
- **监控友好**: 提供详细的分配和预测信息

## 📊 复杂度分析

### 时间复杂度
- **阶段1**: `O(B_max * log(T_max))` 其中B_max为最大batch搜索范围，T_max为token搜索范围
- **阶段2**: `O(R * log(R))` 其中R为running队列大小
- **阶段3**: `O(W * log(W))` 其中W为waiting队列大小

### 总复杂度
`O(B_max * log(T_max) + R*log(R) + W*log(W))`

在典型场景下：
- `B_max ≈ 32`（最大并发数）
- `T_max ≈ 8192`（最大token数）
- `R, W ≤ 100`（队列大小）

总计算量约 `32 * 13 + 100 * 7 ≈ 1100`次操作，**非常高效**！

## 🎯 实际应用场景

### 1. SLA保证
```python
# 为P99延迟优化调度
target_latency = 50.0  # ms
schedule = optimizer.optimize_schedule(running, waiting, target_latency)

# 应用调度决策
for req_id, chunk_size in schedule.running_chunk_sizes.items():
    schedule_running_request(req_id, chunk_size)

for req_id in schedule.scheduled_waiting_ids:
    chunk_size = schedule.waiting_chunk_sizes[req_id]
    promote_waiting_to_running(req_id, chunk_size)
```

### 2. 负载自适应
```python
# 根据队列长度动态调整目标延迟
queue_length = len(waiting_requests)
if queue_length > 10:
    target_latency = 30.0  # 高负载下降低延迟
else:
    target_latency = 50.0  # 低负载下保证质量

schedule = optimizer.optimize_schedule(running, waiting, target_latency)
```

### 3. 多目标优化
```python
# 同时优化延迟和吞吐量
schedules = []
for target_lat in [20, 30, 40, 50]:
    schedule = optimizer.optimize_schedule(running, waiting, target_lat)
    schedules.append((schedule, target_lat))

# 选择帕累托最优解
best_schedule = choose_pareto_optimal(schedules, latency_weight=0.7, throughput_weight=0.3)
```

## 🔍 与原算法对比

| 特性 | 原算法 | 真实场景算法 |
|------|--------|-------------|
| **问题建模** | 抽象化的批次优化 | 基于真实queue结构 |
| **优先级处理** | 简单权重 | 严格的三级优先级 |
| **约束建模** | 理想化约束 | 完整的实际约束 |
| **输出格式** | 通用chunk分配 | 直接对应调度决策 |
| **部署难度** | 需要适配 | 即插即用 |
| **性能** | O(log T + N log N) | O(B log T + R log R + W log W) |

## 🚀 工程实现要点

### 1. 与现有系统集成
```python
class SchedulerWithOptimizer(Scheduler):
    def __init__(self, ...):
        super().__init__(...)
        self.optimizer = RealSchedulerOptimizer(model_params, config)
    
    def schedule(self) -> SchedulerOutput:
        # 使用优化器生成调度方案
        if self.enable_optimization:
            target_latency = self.compute_target_latency()
            optimal_schedule = self.optimizer.optimize_schedule(
                self.running, self.waiting, target_latency
            )
            return self.apply_optimal_schedule(optimal_schedule)
        else:
            # 回退到原始调度逻辑
            return super().schedule()
```

### 2. 参数自适应
```python
# 根据历史性能动态调整模型参数
def update_model_params(self, actual_latencies, predicted_latencies):
    error = mean_squared_error(actual_latencies, predicted_latencies)
    if error > threshold:
        self.model_params = retrain_model(recent_profiling_data)
        self.optimizer.update_params(self.model_params)
```

### 3. 监控和调试
```python
# 详细的调度决策日志
def log_scheduling_decision(self, schedule, target_latency):
    logger.info(f"Optimization result: target={target_latency:.2f}ms, "
               f"predicted={schedule.predicted_latency:.2f}ms, "
               f"running_decisions={schedule.running_chunk_sizes}, "
               f"new_scheduled={len(schedule.scheduled_waiting_ids)}")
```

这个算法设计**完全符合vLLM的真实调度场景**，提供了高效、准确、可部署的反向优化能力！ 