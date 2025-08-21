"""SLA感知调度器主接口模块

该模块提供SLA感知调度器的统一对外接口，整合性能预测器和优化器，
为主调度器提供简洁易用的API。确保与现有vLLM调度器完全兼容。
"""

import time
from typing import Optional, Dict, Any, List, Tuple
import logging

# 导入vLLM相关类型
from vllm.v1.request import Request

from .config import SLASchedulerConfig
from .performance_predictor import PerformancePredictor
from .optimizer import SLAOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class SLAScheduler:
    """SLA感知调度器主接口
    
    提供对外统一的SLA感知调度接口，封装性能预测器和优化器的复杂性，
    为主调度器提供简单易用的API。支持渐进式部署和错误恢复。
    """
    
    def __init__(self, config: Optional[SLASchedulerConfig] = None):
        """初始化SLA调度器
        
        Args:
            config: SLA调度器配置，None时从环境变量加载
        """
        # 加载配置
        self.config = config or SLASchedulerConfig.from_env()
        
        # 初始化组件
        self.predictor = PerformancePredictor(self.config)
        self.optimizer = SLAOptimizer(self.config, self.predictor)
        
        # 状态管理
        self.enabled = self.config.enabled
        self.initialization_time = time.time()
        
        # 统计信息
        self.stats = {
            'total_schedule_calls': 0,
            'successful_optimizations': 0,
            'fallback_count': 0,
            'total_performance_records': 0,
            'avg_optimization_time_ms': 0,
            'last_optimization_result': None,
        }
        
        # 错误恢复
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._error_recovery_time = 0
        
        if self.enabled:
            logger.info(f"SLA Scheduler initialized successfully: {self.config}")
        else:
            logger.info("SLA Scheduler disabled by configuration")
    
    def compute_schedule_decision(self, 
                                 running_requests: List[Request],
                                 waiting_requests: List[Request],
                                 max_tokens: int,
                                 max_batch_size: int) -> Optional[Dict[str, Any]]:
        """计算SLA感知的调度决策
        
        Args:
            running_requests: 运行中的请求列表
            waiting_requests: 等待中的请求列表
            max_tokens: 最大token数限制
            max_batch_size: 最大batch size限制
            
        Returns:
            调度决策字典，包含：
            - 'allocation': Dict[str, int] - request_id -> token数的分配
            - 'token_budget': int - 总token预算
            - 'target_latency': float - 目标延迟
            - 'prioritize_decode': bool - 是否优先decode
            失败时返回None
        """
        if not self.enabled:
            return None
        
        # 检查是否应该使用后备方案
        if self._should_use_fallback():
            return self._fallback_schedule_decision(max_tokens, max_batch_size)
        
        try:
            # 更新统计信息
            self.stats['total_schedule_calls'] += 1
            
            # 计算自适应目标延迟
            queue_length = len(waiting_requests)
            target_latency = self.optimizer.compute_adaptive_target_latency(queue_length)
            
            if self.predictor.is_ready:
                # 使用优化器计算具体分配
                result = self.optimizer.optimize_schedule(
                    running_requests=running_requests,
                    waiting_requests=waiting_requests,
                    target_latency=target_latency,
                    max_batch_size=max_batch_size,
                    max_tokens=max_tokens
                )
                
                if result:
                    self.stats['successful_optimizations'] += 1
                    self.stats['last_optimization_result'] = {
                        'target_latency': result.target_latency,
                        'predicted_latency': result.predicted_latency,
                        'actual_batch_size': result.actual_batch_size,
                        'token_budget': result.optimal_token_budget,
                        'decode_count': result.decode_count,
                        'prefill_count': result.prefill_count,
                    }
                    
                    # 更新优化时间统计
                    self._update_optimization_time_stats(result.optimization_time_ms)
                    
                    # 重置错误计数
                    self._consecutive_errors = 0
                    
                    if self.config.verbose_logging:
                        logger.debug(f"SLA schedule decision: "
                                   f"target={target_latency:.1f}ms, "
                                   f"predicted={result.predicted_latency:.1f}ms, "
                                   f"budget={result.optimal_token_budget}, "
                                   f"batch={result.actual_batch_size}, "
                                   f"allocation={len(result.allocation)} requests")
                    
                    return {
                        'allocation': result.allocation,
                        'token_budget': result.optimal_token_budget,
                        'target_latency': target_latency,
                        'predicted_latency': result.predicted_latency,
                        'prioritize_decode': result.decode_count > 0
                    }
            
            # 后备方案1：基于目标延迟反向计算
            total_requests = len(running_requests) + min(len(waiting_requests), max_batch_size)
            if total_requests > 0:
                token_budget = self.predictor.solve_for_token_budget(total_requests, target_latency)
                token_budget = min(token_budget, max_tokens)
                token_budget = max(1, token_budget)
                
                if self.config.verbose_logging:
                    logger.debug(f"SLA fallback computation: "
                               f"target={target_latency:.1f}ms, "
                               f"budget={token_budget}, "
                               f"batch={total_requests}")
                
                # 创建简单的分配（平均分配token）
                allocation = {}
                for req in running_requests:
                    allocation[req.request_id] = max(1, token_budget // total_requests)
                for req in waiting_requests[:max_batch_size]:
                    allocation[req.request_id] = max(1, token_budget // total_requests)
                
                return {
                    'allocation': allocation,
                    'token_budget': token_budget,
                    'target_latency': target_latency,
                    'prioritize_decode': False
                }
            
            # 后备方案2：使用系统默认值
            return self._fallback_schedule_decision(max_tokens, max_batch_size)
            
        except Exception as e:
            self._handle_error(e)
            return self._fallback_schedule_decision(max_tokens, max_batch_size)

    
    def should_prioritize_decode(self, running_requests: List[Request]) -> bool:
        """判断是否应该优先处理decode请求
        
        Args:
            running_requests: 运行中的请求列表
            
        Returns:
            是否应该优先处理decode请求
        """
        if not self.enabled or not running_requests:
            return False
        
        try:
            decode_count = sum(1 for req in running_requests 
                             if self._is_decode_phase(req))
            
            # 如果有decode请求，建议优先处理
            return decode_count > 0
            
        except Exception as e:
            if self.config.verbose_logging:
                logger.warning(f"Failed to check decode priority: {e}")
            return False
    
    def record_performance(self, batch_size: int, total_tokens: int, 
                          actual_latency: float) -> None:
        """记录性能数据用于模型更新
        
        Args:
            batch_size: 实际batch size
            total_tokens: 实际处理的总token数
            actual_latency: 实际延迟(ms)
        """
        if not self.enabled:
            return
        
        try:
            self.predictor.add_observation(batch_size, total_tokens, actual_latency)
            self.stats['total_performance_records'] += 1
            
            if self.config.verbose_logging:
                logger.debug(f"Performance recorded: B={batch_size}, "
                           f"S={total_tokens}, T={actual_latency:.2f}ms")
                
        except Exception as e:
            if self.config.verbose_logging:
                logger.warning(f"Failed to record performance: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器完整状态信息"""
        predictor_status = self.predictor.get_status()
        optimizer_stats = self.optimizer.get_stats()
        
        uptime_seconds = time.time() - self.initialization_time
        
        return {
            'enabled': self.enabled,
            'uptime_seconds': uptime_seconds,
            'config': self.config.to_dict(),
            'predictor': predictor_status,
            'optimizer': optimizer_stats,
            'stats': self.stats.copy(),
            'error_state': {
                'consecutive_errors': self._consecutive_errors,
                'in_recovery': self._is_in_error_recovery(),
                'recovery_time_remaining': max(0, self._error_recovery_time - time.time()),
            }
        }
    
    def get_simple_status(self) -> Dict[str, Any]:
        """获取简化的状态信息，用于快速监控"""
        return {
            'enabled': self.enabled,
            'predictor_ready': self.predictor.is_ready,
            'total_predictions': self.predictor.stats['total_predictions'],
            'model_predictions': self.predictor.stats['model_predictions'],
            'successful_optimizations': self.stats['successful_optimizations'],
            'fallback_count': self.stats['fallback_count'],
            'avg_optimization_time_ms': self.stats['avg_optimization_time_ms'],
        }
    
    def reset(self) -> None:
        """重置调度器状态"""
        self.predictor.reset()
        self.optimizer.reset_stats()
        
        self.stats = {
            'total_schedule_calls': 0,
            'successful_optimizations': 0,
            'fallback_count': 0,
            'total_performance_records': 0,
            'avg_optimization_time_ms': 0,
            'last_optimization_result': None,
        }
        
        self._consecutive_errors = 0
        self._error_recovery_time = 0
        
        logger.info("SLA Scheduler reset")
    
    def _is_decode_phase(self, request: Request) -> bool:
        """判断请求是否处于decode阶段"""
        try:
            return request.num_computed_tokens >= request.num_prompt_tokens
        except AttributeError:
            return False
    
    def _fallback_budget_computation(self, max_tokens: int) -> Tuple[int, float]:
        """后备预算计算方案"""
        self.stats['fallback_count'] += 1
        
        # 使用保守的默认值
        token_budget = max_tokens
        target_latency = self.config.slo_tpot_ms
        
        if self.config.verbose_logging:
            logger.info(f"Using fallback budget: {token_budget} tokens, {target_latency}ms")
        
        return token_budget, target_latency
    
    def _fallback_schedule_decision(self, max_tokens: int, max_batch_size: int) -> Dict[str, Any]:
        """后备调度决策方案"""
        self.stats['fallback_count'] += 1
        
        # 使用保守的默认值
        token_budget = max_tokens
        target_latency = self.config.slo_tpot_ms
        
        # 创建简单的分配（所有请求平均分配token）
        allocation = {}
        
        if self.config.verbose_logging:
            logger.info(f"Using fallback schedule: {token_budget} tokens, {target_latency}ms")
        
        return {
            'allocation': allocation,
            'token_budget': token_budget,
            'target_latency': target_latency,
            'prioritize_decode': False
        }
    
    def _should_use_fallback(self) -> bool:
        """判断是否应该使用后备方案"""
        # 如果配置不允许回退，始终尝试使用SLA调度
        if not self.config.fallback_on_error:
            return False
        
        # 如果在错误恢复期间，使用后备方案
        if self._is_in_error_recovery():
            return True
        
        # 如果连续错误过多，暂时使用后备方案
        return self._consecutive_errors >= self._max_consecutive_errors
    
    def _is_in_error_recovery(self) -> bool:
        """检查是否处于错误恢复期"""
        return time.time() < self._error_recovery_time
    
    def _handle_error(self, error: Exception) -> None:
        """处理调度错误"""
        self._consecutive_errors += 1
        
        # 如果错误过多，进入恢复期
        if self._consecutive_errors >= self._max_consecutive_errors:
            recovery_duration = min(60, self._consecutive_errors * 5)  # 最多1分钟
            self._error_recovery_time = time.time() + recovery_duration
            
            logger.warning(f"SLA Scheduler entering error recovery mode for {recovery_duration}s "
                         f"after {self._consecutive_errors} consecutive errors. Latest error: {error}")
        else:
            logger.warning(f"SLA Scheduler error ({self._consecutive_errors}/{self._max_consecutive_errors}): {error}")
        
        self.stats['fallback_count'] += 1
    
    def _update_optimization_time_stats(self, optimization_time_ms: float) -> None:
        """更新优化时间统计"""
        alpha = 0.1  # 指数移动平均权重
        if self.stats['avg_optimization_time_ms'] == 0:
            self.stats['avg_optimization_time_ms'] = optimization_time_ms
        else:
            self.stats['avg_optimization_time_ms'] = (
                alpha * optimization_time_ms + 
                (1 - alpha) * self.stats['avg_optimization_time_ms']
            )
    
    def enable(self) -> None:
        """启用SLA调度器"""
        self.enabled = True
        self._consecutive_errors = 0
        self._error_recovery_time = 0
        logger.info("SLA Scheduler enabled")
    
    def disable(self) -> None:
        """禁用SLA调度器"""
        self.enabled = False
        logger.info("SLA Scheduler disabled")
    
    def update_config(self, new_config: SLASchedulerConfig) -> None:
        """动态更新配置
        
        Args:
            new_config: 新的配置
        """
        old_enabled = self.enabled
        
        self.config = new_config
        self.enabled = new_config.enabled
        
        # 如果启用状态发生变化，记录日志
        if old_enabled != self.enabled:
            if self.enabled:
                logger.info("SLA Scheduler enabled by config update")
            else:
                logger.info("SLA Scheduler disabled by config update")
        
        logger.info(f"SLA Scheduler config updated: {new_config}")

    def get_p_max(self) -> Optional[float]:
        """返回吞吐模型的 P_max 参数（tokens/ms）；若不可用返回 None"""
        try:
            return self.predictor.get_p_max()
        except Exception:
            return None
