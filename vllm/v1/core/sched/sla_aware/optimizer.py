"""三阶段SLA感知优化算法模块

该模块实现了论文中描述的三阶段SLA感知调度优化算法：
1. 全局配置优化：搜索最优batch size
2. 运行队列调度：优先处理decode请求，合理分配prefill请求
3. 等待队列选择：基于优先级选择新请求

算法设计确保与现有vLLM调度器完全兼容，同时提供SLA感知的调度决策。
"""

import time
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

# 导入vLLM相关类型
from vllm.v1.request import Request

from .performance_predictor import PerformancePredictor
from .config import SLASchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据类
    
    包含优化算法的所有输出结果，用于调度器进行资源分配决策。
    """
    optimal_batch_size: int                 # 最优batch size
    optimal_token_budget: int              # 最优token预算
    allocation: Dict[str, int]             # request_id -> tokens的分配映射
    predicted_latency: float               # 预测延迟
    optimization_time_ms: float            # 优化算法执行时间
    target_latency: float                  # 目标延迟
    actual_batch_size: int                 # 实际分配的batch size
    decode_count: int                      # decode请求数量
    prefill_count: int                     # prefill请求数量


class SLAOptimizer:
    """SLA感知三阶段优化器
    
    实现论文中描述的延迟导向统一调度算法，通过三个阶段的优化
    实现最佳的资源分配和SLA保证。
    """
    
    def __init__(self, config: SLASchedulerConfig, predictor: PerformancePredictor):
        """初始化优化器
        
        Args:
            config: SLA调度器配置
            predictor: 性能预测器实例
        """
        self.config = config
        self.predictor = predictor
        
        # 优化统计信息
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'timeout_count': 0,
            'avg_optimization_time_ms': 0.0,
        }
    
    def optimize_schedule(self, 
                         running_requests: List[Request],
                         waiting_requests: List[Request],
                         target_latency: float,
                         max_batch_size: int,
                         max_tokens: int) -> Optional[OptimizationResult]:
        """三阶段SLA感知调度优化
        
        实现Algorithm 1: Latency-Guided Unified Scheduling
        
        Args:
            running_requests: 运行中的请求列表
            waiting_requests: 等待中的请求列表
            target_latency: 目标延迟(ms)
            max_batch_size: 最大batch size限制
            max_tokens: 最大token数限制
            
        Returns:
            优化结果，失败时返回None
        """
        start_time = time.perf_counter()
        self.stats['total_optimizations'] += 1
        
        try:
            # 限制搜索范围以确保实时性
            batch_search_limit = len(running_requests) + len(waiting_requests)
            min_batch_size = max(1, len(running_requests))
            
            best_result = None
            min_error = float('inf')
            
            if self.config.verbose_logging:
                logger.debug(f"Starting optimization: target={target_latency:.1f}ms, "
                           f"running={len(running_requests)}, waiting={len(waiting_requests)}")
            
            # Phase 1: 穷举搜索batch size并为每个候选配置执行贪心调度
            for batch_size in range(min_batch_size, batch_search_limit + 1):
                # 超时检查
                if (time.perf_counter() - start_time) * 1000 > self.config.optimization_timeout_ms:
                    self.stats['timeout_count'] += 1
                    if self.config.verbose_logging:
                        logger.info(f"Optimization timeout at batch_size={batch_size}")
                    break
                
                # Phase 2: 求解最优token数
                optimal_tokens = self.predictor.solve_for_token_budget(batch_size, target_latency)
                optimal_tokens = min(optimal_tokens, max_tokens)
                
                if optimal_tokens <= 0:
                    continue
                
                # Phase 3: 贪心分配资源
                allocation = self._greedy_allocation(
                    running_requests, waiting_requests, 
                    batch_size, optimal_tokens
                )
                
                if not allocation:
                    continue
                
                # 验证分配结果
                scheduled_requests = [req_id for req_id, tokens in allocation.items() if tokens > 0]
                actual_batch_size = len(scheduled_requests)
                actual_tokens = sum(allocation.values())
                
                # 确保分配结果满足约束
                if actual_batch_size <= batch_size and actual_tokens <= optimal_tokens:
                    # 预测实际延迟
                    predicted_latency = self.predictor.predict_latency(actual_batch_size, actual_tokens)
                    error = abs(predicted_latency - target_latency)
                    
                    # 更新最佳结果
                    if error < min_error:
                        min_error = error
                        decode_count, prefill_count = self._count_request_types(
                            running_requests, waiting_requests, allocation
                        )
                        
                        best_result = OptimizationResult(
                            optimal_batch_size=batch_size,
                            optimal_token_budget=optimal_tokens,
                            allocation=allocation,
                            predicted_latency=predicted_latency,
                            optimization_time_ms=0,  # 稍后设置
                            target_latency=target_latency,
                            actual_batch_size=actual_batch_size,
                            decode_count=decode_count,
                            prefill_count=prefill_count
                        )
                
                # 如果已经找到了足够好的解，可以提前退出
                if min_error < target_latency * 0.1:  # 10%的误差范围内
                    break
            
            # 设置优化时间
            optimization_time_ms = (time.perf_counter() - start_time) * 1000
            
            if best_result:
                best_result.optimization_time_ms = optimization_time_ms
                self.stats['successful_optimizations'] += 1
                self._update_stats(optimization_time_ms)
                
                if self.config.verbose_logging:
                    logger.debug(f"Optimization success: B={best_result.actual_batch_size}, "
                               f"S={sum(best_result.allocation.values())}, "
                               f"T_pred={best_result.predicted_latency:.1f}ms, "
                               f"error={min_error:.1f}ms, time={optimization_time_ms:.2f}ms")
                
                return best_result
            else:
                if self.config.verbose_logging:
                    logger.info(f"Optimization failed: no valid allocation found")
                
                return None
            
        except Exception as e:
            logger.error(f"Optimization failed with exception: {e}")
            return None
    
    def _greedy_allocation(self, 
                          running_requests: List[Request],
                          waiting_requests: List[Request],
                          batch_size: int,
                          token_budget: int) -> Dict[str, int]:
        """贪心分配算法
        
        实现论文中描述的三阶段贪心策略：
        1. Running中的decode请求（每个需要1 token）
        2. Running中的prefill请求
        3. Waiting中的新请求（按优先级排序）
        
        Args:
            running_requests: 运行中的请求
            waiting_requests: 等待中的请求
            batch_size: 目标batch size
            token_budget: token预算
            
        Returns:
            request_id -> tokens的分配字典
        """
        allocation = {}
        remaining_budget = token_budget
        remaining_slots = batch_size
        
        # Phase 1: 分类running请求
        decode_requests = []
        prefill_requests = []
        
        for req in running_requests:
            if self._is_decode_phase(req):
                decode_requests.append(req)
            else:
                prefill_requests.append(req)
        
        # Phase 2: 优先分配decode请求（每个1 token）
        for req in decode_requests:
            if remaining_budget >= 1 and remaining_slots > 0:
                allocation[req.request_id] = 1
                remaining_budget -= 1
                remaining_slots -= 1
            else:
                allocation[req.request_id] = 0
        
        # Phase 3: 分配running prefill请求
        # 按剩余token数排序，优先处理即将完成的请求
        prefill_requests.sort(key=lambda req: self._get_remaining_prefill_tokens(req))
        
        for req in prefill_requests:
            if remaining_budget <= 0 or remaining_slots <= 0:
                allocation[req.request_id] = 0
                continue
            
            remaining_tokens = self._get_remaining_prefill_tokens(req)
            # 限制chunk大小以避免过度占用资源
            max_chunk = min(remaining_tokens, 512)  # 最大chunk限制
            chunk_size = min(max_chunk, remaining_budget)
            
            allocation[req.request_id] = chunk_size
            remaining_budget -= chunk_size
            remaining_slots -= 1
        
        # Phase 4: 选择waiting请求
        if remaining_budget > 0 and remaining_slots > 0:
            selected_waiting = self._select_waiting_requests(
                waiting_requests, remaining_slots, remaining_budget
            )
            
            for req, tokens in selected_waiting:
                allocation[req.request_id] = tokens
                remaining_budget -= tokens
                remaining_slots -= 1
        
        return allocation
    
    def _select_waiting_requests(self, 
                               waiting_requests: List[Request],
                               remaining_slots: int,
                               remaining_budget: int) -> List[Tuple[Request, int]]:
        """选择等待队列中的请求
        
        Args:
            waiting_requests: 等待中的请求列表
            remaining_slots: 剩余slot数
            remaining_budget: 剩余token预算
            
        Returns:
            选中的请求及其token分配列表
        """
        if not waiting_requests or remaining_slots <= 0 or remaining_budget <= 0:
            return []
        
        # 按优先级排序（如果有优先级字段）
        # 注意：vLLM Request可能没有priority字段，需要兼容处理
        try:
            sorted_waiting = sorted(waiting_requests, 
                                  key=lambda req: getattr(req, 'priority', 0), 
                                  reverse=True)
        except AttributeError:
            # 如果没有priority字段，按FIFO顺序
            sorted_waiting = waiting_requests
        
        selected = []
        
        for req in sorted_waiting:
            if remaining_slots <= 0 or remaining_budget <= 0:
                break
            
            # 计算启动该请求需要的最小token数
            prompt_tokens = getattr(req, 'num_prompt_tokens', 0)
            if prompt_tokens <= 0:
                # 如果无法获取prompt长度，使用默认最小值
                min_startup_tokens = 16
            else:
                min_startup_tokens = min(16, prompt_tokens)
            
            if remaining_budget < min_startup_tokens:
                # 剩余预算不足以启动新请求
                break
            
            # 计算该请求的token分配
            max_chunk = min(prompt_tokens, 512) if prompt_tokens > 0 else 256
            chunk_size = min(max_chunk, remaining_budget)
            
            if chunk_size >= min_startup_tokens:
                selected.append((req, chunk_size))
                remaining_budget -= chunk_size
                remaining_slots -= 1
        
        return selected
    
    def _is_decode_phase(self, request: Request) -> bool:
        """判断请求是否处于decode阶段"""
        try:
            return request.num_computed_tokens >= request.num_prompt_tokens
        except AttributeError:
            # 如果字段不存在，假设是prefill阶段
            return False
    
    def _get_remaining_prefill_tokens(self, request: Request) -> int:
        """获取prefill请求的剩余token数"""
        try:
            return max(0, request.num_prompt_tokens - request.num_computed_tokens)
        except AttributeError:
            # 如果字段不存在，返回默认值
            return 256
    
    def _count_request_types(self, 
                           running_requests: List[Request],
                           waiting_requests: List[Request],
                           allocation: Dict[str, int]) -> Tuple[int, int]:
        """统计分配结果中的decode和prefill请求数量"""
        decode_count = 0
        prefill_count = 0
        
        for req in running_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                if self._is_decode_phase(req):
                    decode_count += 1
                else:
                    prefill_count += 1
        
        for req in waiting_requests:
            if req.request_id in allocation and allocation[req.request_id] > 0:
                prefill_count += 1  # 新请求都是prefill
        
        return decode_count, prefill_count
    
    def compute_adaptive_target_latency(self, queue_length: int) -> float:
        """计算自适应目标延迟
        
        基于队列长度的线性插值计算目标延迟，实现负载感知调度。
        
        Args:
            queue_length: 等待队列长度
            
        Returns:
            目标延迟(ms)
        """
        # 基于队列长度的线性插值
        if self.config.queue_threshold > 0:
            k = (self.config.slo_tpot_ms - self.config.min_batch_time_ms) / self.config.queue_threshold
            target = self.config.min_batch_time_ms + k * queue_length
        else:
            target = self.config.min_batch_time_ms
        
        # 确保有效下界：至少能处理1个token
        min_effective = max(
            self.config.min_batch_time_ms,
            self.predictor.fallback_intercept + self.predictor.fallback_slope
        )
        
        adaptive_target = max(min_effective, min(self.config.slo_tpot_ms, target))
        
        if self.config.verbose_logging:
            logger.debug(f"Adaptive target latency: Q={queue_length} -> {adaptive_target:.1f}ms")
        
        return adaptive_target
    
    def _update_stats(self, optimization_time_ms: float) -> None:
        """更新优化器统计信息"""
        # 计算移动平均
        alpha = 0.1  # 指数移动平均的权重
        if self.stats['avg_optimization_time_ms'] == 0:
            self.stats['avg_optimization_time_ms'] = optimization_time_ms
        else:
            self.stats['avg_optimization_time_ms'] = (
                alpha * optimization_time_ms + 
                (1 - alpha) * self.stats['avg_optimization_time_ms']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        total = self.stats['total_optimizations']
        success_rate = (self.stats['successful_optimizations'] / total) if total > 0 else 0.0
        
        return {
            'total_optimizations': total,
            'successful_optimizations': self.stats['successful_optimizations'],
            'success_rate': success_rate,
            'timeout_count': self.stats['timeout_count'],
            'avg_optimization_time_ms': self.stats['avg_optimization_time_ms'],
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'timeout_count': 0,
            'avg_optimization_time_ms': 0.0,
        }
