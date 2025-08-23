#!/usr/bin/env python3
"""
vLLM 真实调度场景的反向优化算法

基于scheduler.py的实际调度逻辑，设计符合running/waiting队列结构的反向优化算法。
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import logging
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RequestPhase(Enum):
    """请求阶段"""
    DECODE = "decode"           # 在decode阶段的running请求
    CHUNKED_PREFILL = "chunked_prefill"  # 在chunked prefill阶段的running请求
    NEW_PREFILL = "new_prefill" # waiting队列中的新请求


@dataclass
class RealRequestInfo:
    """真实的请求信息（基于scheduler.py）"""
    request_id: str
    phase: RequestPhase
    
    # 基本token信息
    num_prompt_tokens: int           # 总prompt长度
    num_computed_tokens: int         # 已计算的token数
    num_cached_tokens: int           # 缓存命中的token数
    
    # 调度相关
    priority: float = 1.0            # 优先级权重
    max_chunk_size: int = 512        # 最大chunk限制
    
    @property
    def remaining_prompt_tokens(self) -> int:
        """剩余需要prefill的token数"""
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)
    
    @property
    def is_decode_phase(self) -> bool:
        """是否处于decode阶段"""
        return self.num_computed_tokens >= self.num_prompt_tokens


@dataclass
class SchedulingConfig:
    """调度配置"""
    max_num_scheduled_tokens: int   # 最大token预算
    max_num_running_reqs: int       # 最大并发请求数
    long_prefill_threshold: int = 2048  # 长prefill分块阈值
    enable_load_aware: bool = False  # 是否启用负载感知
    

@dataclass 
class OptimalSchedule:
    """最优调度结果"""
    # 调度决策
    running_chunk_sizes: Dict[str, int]      # running请求的chunk分配
    scheduled_waiting_ids: List[str]         # 被调度的waiting请求ID
    waiting_chunk_sizes: Dict[str, int]      # waiting请求的chunk分配
    
    # 预测结果
    total_tokens: int                        # 总token数
    predicted_latency: float                 # 预测延迟
    
    # 统计信息
    num_decode_tokens: int                   # decode token数
    num_prefill_tokens: int                  # prefill token数
    feasible: bool = True                    # 是否可行


class RealSchedulerOptimizer:
    """真实调度场景的反向优化器"""
    
    def __init__(self, model_params: Dict, config: SchedulingConfig, verbose: bool = False):
        """
        初始化优化器
        
        Args:
            model_params: 性能模型参数
            config: 调度配置
            verbose: 是否输出详细信息
        """
        self.model_params = model_params
        self.config = config
        self.verbose = verbose
        
        # 提取模型参数
        self.P_max = model_params['P_max']
        self.k_B = model_params['k_B'] 
        self.k_S = model_params['k_S']
        self.w_0 = model_params['w_0']
        self.w_1 = model_params['w_1']
        self.tau_0 = model_params['tau_0']
        self.tau_B = model_params['tau_B']
        self.tau_S = model_params['tau_S']
        
        if verbose:
            logger.info("真实调度优化器已初始化")
    
    def _latency_function(self, batch_size: int, total_tokens: int) -> float:
        """计算延迟"""
        if total_tokens <= 0 or batch_size <= 0:
            return float('inf')
            
        # 吞吐量计算
        thr = self.P_max * (1 - np.exp(-self.k_B * batch_size)) * (1 - np.exp(-self.k_S * total_tokens))
        if thr <= 1e-9:
            return float('inf')
            
        # 工作量计算  
        work = self.w_0 + self.w_1 * total_tokens
        
        # 总延迟
        latency = self.tau_0 + work / thr + self.tau_B * batch_size + self.tau_S * total_tokens
        
        return latency
    
    def _solve_optimal_tokens(self, batch_size: int, target_latency: float) -> Optional[int]:
        """给定batch_size求解最优total_tokens"""
        def latency_diff(S):
            return self._latency_function(batch_size, int(S)) - target_latency
        
        # 搜索区间
        S_min = batch_size  # 最小：全decode
        S_max = min(self.config.max_num_scheduled_tokens, batch_size * 2048)
        
        # 边界检查
        if latency_diff(S_min) > 0:
            if self.verbose:
                logger.warning(f"目标延迟过低: {target_latency:.2f}ms")
            return None
            
        if latency_diff(S_max) < 0:
            if self.verbose:
                logger.warning(f"目标延迟过高: {target_latency:.2f}ms")
            return S_max
        
        # 求解
        try:
            S_optimal = brentq(latency_diff, S_min, S_max, xtol=1e-3)
            return int(np.round(S_optimal))
        except ValueError as e:
            if self.verbose:
                logger.error(f"求解失败: {e}")
            return None
    
    def _schedule_running_requests(self, running_requests: List[RealRequestInfo], 
                                 available_tokens: int) -> Tuple[Dict[str, int], int]:
        """
        调度running队列请求（优先级最高）
        
        策略：
        1. decode请求优先，每个分配1个token
        2. 再调度chunked prefill，按剩余需求分配
        
        Returns:
            (chunk_sizes, remaining_tokens)
        """
        chunk_sizes = {}
        remaining_tokens = available_tokens
        
        # 分离decode和prefill请求
        decode_reqs = [req for req in running_requests if req.is_decode_phase]
        prefill_reqs = [req for req in running_requests if not req.is_decode_phase]
        
        # 1. 优先调度decode请求
        for req in decode_reqs:
            if remaining_tokens >= 1:
                chunk_sizes[req.request_id] = 1
                remaining_tokens -= 1
            else:
                chunk_sizes[req.request_id] = 0
        
        # 2. 调度running prefill请求
        if prefill_reqs and remaining_tokens > 0:
            # 按优先级和剩余需求排序
            prefill_reqs.sort(key=lambda r: (-r.priority, -r.remaining_prompt_tokens))
            
            for req in prefill_reqs:
                if remaining_tokens <= 0:
                    chunk_sizes[req.request_id] = 0
                    continue
                
                # 计算可分配的token数
                max_needed = req.remaining_prompt_tokens
                max_chunk = min(req.max_chunk_size, self.config.long_prefill_threshold)
                allocation = min(max_needed, max_chunk, remaining_tokens)
                
                chunk_sizes[req.request_id] = allocation
                remaining_tokens -= allocation
        
        return chunk_sizes, remaining_tokens
    
    def _select_waiting_requests(self, waiting_requests: List[RealRequestInfo],
                               available_tokens: int, available_slots: int) -> Tuple[List[str], Dict[str, int], int]:
        """
        从waiting队列选择请求进行调度
        
        策略：
        1. 按优先级排序
        2. 贪心选择，确保每个请求至少能分配最小chunk
        
        Returns:
            (selected_ids, chunk_sizes, remaining_tokens)
        """
        if not waiting_requests or available_tokens <= 0 or available_slots <= 0:
            return [], {}, available_tokens
        
        # 按优先级排序
        sorted_waiting = sorted(waiting_requests, key=lambda r: -r.priority)
        
        selected_ids = []
        chunk_sizes = {}
        remaining_tokens = available_tokens
        remaining_slots = available_slots
        
        for req in sorted_waiting:
            if remaining_slots <= 0 or remaining_tokens <= 0:
                break
            
            # 计算初始分配
            min_chunk = min(64, req.num_prompt_tokens)  # 最小chunk大小
            max_chunk = min(req.max_chunk_size, self.config.long_prefill_threshold)
            
            if remaining_tokens < min_chunk:
                break  # 剩余token不足以启动新请求
            
            # 分配token
            allocation = min(req.num_prompt_tokens, max_chunk, remaining_tokens)
            
            selected_ids.append(req.request_id)
            chunk_sizes[req.request_id] = allocation
            remaining_tokens -= allocation
            remaining_slots -= 1
        
        return selected_ids, chunk_sizes, remaining_tokens
    
    def optimize_schedule(self, running_requests: List[RealRequestInfo],
                         waiting_requests: List[RealRequestInfo],
                         target_latency: float) -> OptimalSchedule:
        """
        主优化算法：为给定目标延迟优化调度决策
        
        算法流程：
        1. 估算合适的batch_size和total_tokens
        2. 优先调度running请求
        3. 贪心选择waiting请求
        4. 精细调整以逼近目标延迟
        """
        if self.verbose:
            logger.info(f"开始调度优化: running={len(running_requests)}, "
                       f"waiting={len(waiting_requests)}, target={target_latency:.2f}ms")
        
        # 当前running队列大小
        current_batch_size = len(running_requests)
        
        # 尝试不同的batch_size配置
        best_schedule = None
        best_error = float('inf')
        
        # 搜索范围：当前batch_size到最大并发数
        max_batch_size = min(
            current_batch_size + len(waiting_requests),
            self.config.max_num_running_reqs
        )
        
        for target_batch_size in range(current_batch_size, max_batch_size + 1):
            # 求解该batch_size下的最优token数
            optimal_tokens = self._solve_optimal_tokens(target_batch_size, target_latency)
            if optimal_tokens is None:
                continue
            
            # 限制在预算内
            optimal_tokens = min(optimal_tokens, self.config.max_num_scheduled_tokens)
            
            # 调度running请求
            running_chunks, remaining_tokens = self._schedule_running_requests(
                running_requests, optimal_tokens
            )
            
            # 调度waiting请求
            available_slots = target_batch_size - current_batch_size
            selected_waiting, waiting_chunks, final_remaining = self._select_waiting_requests(
                waiting_requests, remaining_tokens, available_slots
            )
            
            # 计算实际配置
            actual_batch_size = current_batch_size + len(selected_waiting)
            actual_tokens = optimal_tokens - final_remaining
            actual_latency = self._latency_function(actual_batch_size, actual_tokens)
            
            # 评估误差
            error = abs(actual_latency - target_latency)
            
            if error < best_error:
                best_error = error
                best_schedule = OptimalSchedule(
                    running_chunk_sizes=running_chunks,
                    scheduled_waiting_ids=selected_waiting,
                    waiting_chunk_sizes=waiting_chunks,
                    total_tokens=actual_tokens,
                    predicted_latency=actual_latency,
                    num_decode_tokens=sum(1 for req in running_requests if req.is_decode_phase),
                    num_prefill_tokens=actual_tokens - sum(1 for req in running_requests if req.is_decode_phase),
                    feasible=True
                )
        
        if best_schedule is None:
            # 返回保守方案
            running_chunks, _ = self._schedule_running_requests(
                running_requests, self.config.max_num_scheduled_tokens
            )
            
            best_schedule = OptimalSchedule(
                running_chunk_sizes=running_chunks,
                scheduled_waiting_ids=[],
                waiting_chunk_sizes={},
                total_tokens=sum(running_chunks.values()),
                predicted_latency=self._latency_function(len(running_requests), sum(running_chunks.values())),
                num_decode_tokens=sum(1 for req in running_requests if req.is_decode_phase),
                num_prefill_tokens=sum(running_chunks.values()) - sum(1 for req in running_requests if req.is_decode_phase),
                feasible=False
            )
        
        if self.verbose:
            logger.info(f"优化完成: batch_size={len(running_requests) + len(best_schedule.scheduled_waiting_ids)}, "
                       f"tokens={best_schedule.total_tokens}, "
                       f"latency={best_schedule.predicted_latency:.2f}ms, "
                       f"error={abs(best_schedule.predicted_latency - target_latency):.2f}ms")
        
        return best_schedule


def create_real_scenario_requests() -> Tuple[List[RealRequestInfo], List[RealRequestInfo]]:
    """创建真实场景的请求示例"""
    
    # Running请求：混合decode和chunked prefill
    running_requests = [
        # Decode阶段的请求
        RealRequestInfo("run_1", RequestPhase.DECODE, 512, 512, 256),
        RealRequestInfo("run_2", RequestPhase.DECODE, 256, 256, 128),
        RealRequestInfo("run_3", RequestPhase.DECODE, 1024, 1024, 512),
        
        # Chunked prefill阶段的请求
        RealRequestInfo("run_4", RequestPhase.CHUNKED_PREFILL, 2048, 256, 128),
        RealRequestInfo("run_5", RequestPhase.CHUNKED_PREFILL, 1536, 512, 256),
    ]
    
    # Waiting请求：都是新的prefill请求
    waiting_requests = [
        RealRequestInfo("wait_1", RequestPhase.NEW_PREFILL, 1024, 0, 0, priority=2.0),
        RealRequestInfo("wait_2", RequestPhase.NEW_PREFILL, 512, 0, 0, priority=1.5),
        RealRequestInfo("wait_3", RequestPhase.NEW_PREFILL, 2048, 0, 0, priority=1.0),
        RealRequestInfo("wait_4", RequestPhase.NEW_PREFILL, 768, 0, 0, priority=3.0),
    ]
    
    return running_requests, waiting_requests


def demo_real_scheduler_optimization():
    """演示真实调度场景的优化"""
    print("🚀 vLLM 真实调度场景优化演示")
    print("=" * 60)
    
    # 模型参数
    model_params = {
        'P_max': 50.0,
        'k_B': 0.05,
        'k_S': 0.005,
        'w_0': 2.0,
        'w_1': 0.05,
        'tau_0': 5.0,
        'tau_B': 0.5,
        'tau_S': 0.01
    }
    
    # 调度配置
    config = SchedulingConfig(
        max_num_scheduled_tokens=4096,
        max_num_running_reqs=32,
        long_prefill_threshold=512
    )
    
    # 创建优化器
    optimizer = RealSchedulerOptimizer(model_params, config, verbose=True)
    
    # 创建测试场景
    running_reqs, waiting_reqs = create_real_scenario_requests()
    
    print(f"\n📊 当前状态:")
    print(f"Running队列: {len(running_reqs)}个请求")
    for req in running_reqs:
        status = "Decode" if req.is_decode_phase else f"Prefill({req.remaining_prompt_tokens} left)"
        print(f"  - {req.request_id}: {status}")
    
    print(f"Waiting队列: {len(waiting_reqs)}个请求")
    for req in waiting_reqs:
        print(f"  - {req.request_id}: Prefill({req.num_prompt_tokens} tokens), Priority={req.priority}")
    
    # 测试不同目标延迟
    target_latencies = [30.0, 45.0, 60.0]
    
    for target_lat in target_latencies:
        print(f"\n🎯 目标延迟: {target_lat}ms")
        print("-" * 40)
        
        schedule = optimizer.optimize_schedule(running_reqs, waiting_reqs, target_lat)
        
        if schedule.feasible:
            print(f"✅ 优化成功!")
            print(f"   预测延迟: {schedule.predicted_latency:.2f}ms")
            print(f"   误差: {abs(schedule.predicted_latency - target_lat):.2f}ms")
            print(f"   总Token数: {schedule.total_tokens}")
            print(f"   新调度请求: {len(schedule.scheduled_waiting_ids)}个")
            
            # 详细分配信息
            print(f"   Running请求分配:")
            for req_id, chunk_size in schedule.running_chunk_sizes.items():
                print(f"     {req_id}: {chunk_size} tokens")
            
            if schedule.scheduled_waiting_ids:
                print(f"   新调度的Waiting请求:")
                for req_id in schedule.scheduled_waiting_ids:
                    chunk_size = schedule.waiting_chunk_sizes.get(req_id, 0)
                    print(f"     {req_id}: {chunk_size} tokens")
        else:
            print(f"❌ 无法满足目标延迟，采用保守方案")
            print(f"   预测延迟: {schedule.predicted_latency:.2f}ms")


if __name__ == '__main__':
    demo_real_scheduler_optimization() 