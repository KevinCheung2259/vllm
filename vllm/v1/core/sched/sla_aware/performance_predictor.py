"""性能预测器模块

该模块提供基于吞吐饱和理论的性能预测功能，集成本地的ThroughputSaturationModel，
并提供调度器友好的接口。支持预拟合模型加载和在线模型更新。
"""

import sys
import os
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any, Tuple, List, Union
import time
import pandas as pd
import numpy as np
import logging

# 导入本地的吞吐饱和模型
try:
    from .throughput_model import ThroughputSaturationModel, StableClusterModel
    MODEL_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"ThroughputSaturationModel import failed: {e}")
    ThroughputSaturationModel = None
    StableClusterModel = None
    MODEL_AVAILABLE = False

try:
    from .config import SLASchedulerConfig
except ImportError:
    from config import SLASchedulerConfig

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """性能预测器
    
    支持三种工作模式：
    1. 预拟合模型：直接加载已训练好的模型文件
    2. 在线训练：基于实时数据训练模型
    3. 线性后备：当吞吐饱和模型不可用时使用简单线性模型
    """
    
    def __init__(self, config: SLASchedulerConfig):
        """初始化性能预测器
        
        Args:
            config: SLA调度器配置
        """
        self.config = config
        self.model: Optional[Union[ThroughputSaturationModel, 'StableClusterModel']] = None
        self.performance_buffer: deque = deque(maxlen=config.max_buffer_size)
        self.last_update_time = 0
        self.is_ready = False
        self._update_count = 0
        
        # 线性后备模型参数
        self.fallback_intercept = config.fallback_intercept_ms  # 截距
        self.fallback_slope = config.fallback_slope_ms_per_token  # 斜率
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'model_predictions': 0,
            'fallback_predictions': 0,
            'last_mape': float('inf'),
            'last_r2': 0.0,
        }
        
        # 初始化模型
        self._initialize_model()
        
        logger.info(f"PerformancePredictor initialized with config: {config}")
    
    def _initialize_model(self):
        """初始化模型：尝试加载预拟合模型或准备在线训练"""
        if not MODEL_AVAILABLE:
            logger.warning("ThroughputSaturationModel not available, using linear fallback only")
            return
        
        # 添加详细的初始化日志
        if self.config.verbose_logging:
            logger.info(f"Initializing model: use_stable={self.config.use_stable_cluster_model}, "
                       f"use_pretrained={self.config.use_pretrained_model}, "
                       f"pretrained_path='{self.config.pretrained_model_path}'")
        
        # 尝试加载预拟合模型
        if self.config.use_pretrained_model and self.config.pretrained_model_path:
            try:
                model_path = self.config.pretrained_model_path
                
                # 如果是相对路径，尝试相对于当前代码文件的路径
                if not os.path.isabs(model_path):
                    # 获取当前代码文件所在目录
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(current_dir, model_path)
                    if self.config.verbose_logging:
                        logger.info(f"Resolved relative path to: {model_path}")
                
                if os.path.exists(model_path):
                    # 根据配置选择模型类型
                    if self.config.use_stable_cluster_model:
                        self.model = StableClusterModel(verbose=self.config.verbose_logging)
                        model_type = "StableClusterModel"
                    else:
                        self.model = ThroughputSaturationModel(verbose=self.config.verbose_logging)
                        model_type = "ThroughputSaturationModel"
                    
                    self.model.load_model(model_path)
                    self.is_ready = True
                    self._update_count = 1  # 标记为已有模型
                    
                    logger.info(f"Loaded pretrained {model_type} from: {model_path}")
                    logger.info(f"Model R²: {self.model.fit_metrics.get('r2', 'N/A')}, P_max: {self.model.P_max}")
                    
                    # 更新统计信息
                    self.stats['last_r2'] = self.model.fit_metrics.get('r2', 0.0)
                    return
                else:
                    logger.warning(f"Pretrained model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
        else:
            if self.config.verbose_logging:
                if not self.config.use_pretrained_model:
                    logger.info("Pretrained model disabled by configuration")
                elif not self.config.pretrained_model_path:
                    logger.info("No pretrained model path specified")
        
        # 如果没有预拟合模型，准备在线训练
        if MODEL_AVAILABLE and self.config.enabled:
            if self.config.use_stable_cluster_model:
                self.model = StableClusterModel(verbose=self.config.verbose_logging)
                model_type = "StableClusterModel"
            else:
                self.model = ThroughputSaturationModel(verbose=self.config.verbose_logging)
                model_type = "ThroughputSaturationModel"
            if self.config.verbose_logging:
                logger.info(f"Prepared for online {model_type} training")
    
    def add_observation(self, batch_size: int, total_tokens: int, 
                       actual_latency: float, timestamp: Optional[float] = None) -> None:
        """添加性能观测数据
        
        Args:
            batch_size: 批次大小
            total_tokens: 总token数
            actual_latency: 实际延迟(ms)
            timestamp: 时间戳，None时使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 验证数据合理性
        if batch_size <= 0 or total_tokens <= 0 or actual_latency <= 0:
            if self.config.verbose_logging:
                logger.warning(f"Invalid observation: B={batch_size}, S={total_tokens}, T={actual_latency}")
            return
            
        observation = {
            'batch_size': batch_size,
            'total_tokens': total_tokens,
            'latency': actual_latency,
            'timestamp': timestamp
        }
        
        self.performance_buffer.append(observation)
        
        if self.config.verbose_logging:
            logger.debug(f"Added observation: B={batch_size}, S={total_tokens}, T={actual_latency:.2f}ms")
        
        # 如果使用预拟合模型，不进行在线更新
        if self.config.use_pretrained_model and self.is_ready:
            return
        
        # 检查是否需要更新模型
        if self._should_update_model():
            self._update_model()
    
    def predict_latency(self, batch_size: int, total_tokens: int) -> float:
        """预测延迟
        
        Args:
            batch_size: 批次大小
            total_tokens: 总token数
            
        Returns:
            预测的延迟(ms)
        """
        self.stats['total_predictions'] += 1
        
        try:
            if self.is_ready and self.model is not None:
                # 使用吞吐饱和模型预测
                prediction = float(self.model.predict(batch_size, total_tokens))
                self.stats['model_predictions'] += 1
                
                if self.config.verbose_logging:
                    logger.debug(f"Model prediction: B={batch_size}, S={total_tokens} -> {prediction:.2f}ms")
                
                return max(prediction, 0.1)  # 确保预测值为正
                
            else:
                # 使用线性后备模型
                prediction = self._fallback_predict(batch_size, total_tokens)
                self.stats['fallback_predictions'] += 1
                
                if self.config.verbose_logging:
                    logger.info(f"Fallback prediction: B={batch_size}, S={total_tokens} -> {prediction:.2f}ms")
                
                return prediction
                
        except Exception as e:
            if self.config.verbose_logging:
                logger.warning(f"Prediction failed, using fallback: {e}")
            
            self.stats['fallback_predictions'] += 1
            return self._fallback_predict(batch_size, total_tokens)
    
    def solve_for_token_budget(self, batch_size: int, target_latency: float) -> int:
        """反向求解：给定batch_size和目标延迟，求解token预算
        
        Args:
            batch_size: 批次大小
            target_latency: 目标延迟(ms)
            
        Returns:
            推荐的token预算
        """
        try:
            if self.is_ready and self.model is not None:
                # 使用二分搜索求解
                tokens = self._binary_search_tokens(batch_size, target_latency)
            else:
                # 线性模型反向求解
                tokens = self._fallback_solve_tokens(target_latency)
            
            # 确保结果合理
            tokens = max(1, min(tokens, 4096))  # 限制在合理范围内
            
            if self.config.verbose_logging:
                logger.debug(f"Solved tokens: B={batch_size}, T_target={target_latency:.2f}ms -> {tokens} tokens")
            
            return tokens
            
        except Exception as e:
            if self.config.verbose_logging:
                logger.warning(f"Token budget solving failed, using fallback: {e}")
            
            return self._fallback_solve_tokens(target_latency)
    
    def _fallback_predict(self, batch_size: int, total_tokens: int) -> float:
        """线性后备模型预测"""
        # 简单线性模型：主要基于total_tokens，batch_size的影响较小
        prediction = self.fallback_intercept + self.fallback_slope * total_tokens
        return max(prediction, 0.1)
    
    def _fallback_solve_tokens(self, target_latency: float) -> int:
        """线性模型反向求解token数量"""
        if self.fallback_slope <= 0:
            return 1
        
        tokens = (target_latency - self.fallback_intercept) / self.fallback_slope
        return max(1, int(tokens))
    
    def _binary_search_tokens(self, batch_size: int, target_latency: float, 
                             max_iterations: int = 20) -> int:
        """二分搜索求解token数量"""
        low, high = 1, 4096
        best_tokens = 1
        
        for _ in range(max_iterations):
            mid = (low + high) // 2
            try:
                pred_latency = self.model.predict(batch_size, mid)
                
                if abs(pred_latency - target_latency) < 1:  # 足够接近
                    return mid
                
                if pred_latency < target_latency:
                    low = mid + 1
                    best_tokens = mid
                else:
                    high = mid - 1
                    
            except Exception:
                # 如果预测失败，返回当前最佳值
                break
        
        return best_tokens
    
    def _should_update_model(self) -> bool:
        """判断是否应该更新模型"""
        # 如果使用预拟合模型，不进行更新
        if self.config.use_pretrained_model and self.is_ready:
            return False
        
        if not MODEL_AVAILABLE:
            return False
        
        if len(self.performance_buffer) < self.config.min_samples_for_update:
            return False
        
        # 如果模型未初始化，直接更新
        if not self.is_ready:
            logger.info("Model not initialized, updating model")
            return True
        
        # 检查预测精度是否下降
        recent_data = list(self.performance_buffer)[-self.config.min_samples_for_update:]
        mape = self._calculate_mape(recent_data)
        
        should_update = mape > self.config.model_update_threshold
        
        if should_update and self.config.verbose_logging:
            logger.info(f"Model update triggered: MAPE={mape:.3f} > threshold={self.config.model_update_threshold}")
        
        return should_update
    
    def _calculate_mape(self, data: List[dict]) -> float:
        """计算平均绝对百分比误差"""
        if not self.is_ready or not data:
            return float('inf')
        
        errors = []
        for obs in data:
            try:
                pred = self.predict_latency(obs['batch_size'], obs['total_tokens'])
                actual = obs['latency']
                if actual > 0:
                    errors.append(abs(pred - actual) / actual)
            except Exception:
                continue
        
        mape = np.mean(errors) if errors else float('inf')
        self.stats['last_mape'] = mape
        return mape
    
    def _update_model(self) -> bool:
        """更新性能模型"""
        if not MODEL_AVAILABLE:
            return False
        
        try:
            # 准备数据
            data = list(self.performance_buffer)
            if len(data) < self.config.min_samples_for_update:
                return False
            
            df = pd.DataFrame(data)
            
            # 添加必要的列名映射以兼容ThroughputSaturationModel
            df['model_run_duration_ms'] = df['latency']
            
            # 创建并拟合模型
            if self.model is None:
                self.model = ThroughputSaturationModel(verbose=self.config.verbose_logging)
            
            self.model.fit(df)
            
            # 验证模型质量
            r2 = self.model.fit_metrics.get('r2', 0.0) if hasattr(self.model, 'fit_metrics') else 0.0
            
            if r2 > self.config.model_confidence_threshold:
                self.is_ready = True
                self.last_update_time = time.time()
                self._update_count += 1
                self.stats['last_r2'] = r2
                
                # 保存模型（如果配置允许）
                if self.config.save_trained_model and self.config.model_save_path:
                    try:
                        self.model.save_model(self.config.model_save_path)
                        if self.config.verbose_logging:
                            logger.info(f"Model saved to: {self.config.model_save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save model: {e}")
                
                if self.config.verbose_logging:
                    logger.info(f"Model updated successfully (#{self._update_count}): R²={r2:.3f}, samples={len(data)}")
                
                return True
            else:
                if self.config.verbose_logging:
                    logger.warning(f"Model quality too low: R²={r2:.3f} < threshold={self.config.model_confidence_threshold}")
                
                return False
                
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取预测器状态"""
        return {
            'is_ready': self.is_ready,
            'buffer_size': len(self.performance_buffer),
            'max_buffer_size': self.config.max_buffer_size,
            'last_update_time': self.last_update_time,
            'update_count': self._update_count,
            'model_r2': self.stats['last_r2'],
            'last_mape': self.stats['last_mape'],
            'stats': self.stats.copy(),
            'model_available': MODEL_AVAILABLE,
            'using_pretrained': self.config.use_pretrained_model and self.is_ready,
            'pretrained_path': self.config.pretrained_model_path if self.config.use_pretrained_model else None,
        }
    
    def reset(self) -> None:
        """重置预测器状态"""
        self.performance_buffer.clear()
        
        # 如果使用预拟合模型，保持模型不变
        if not (self.config.use_pretrained_model and self.is_ready):
            self.model = None
            self.is_ready = False
            self._update_count = 0
        
        self.stats = {
            'total_predictions': 0,
            'model_predictions': 0,
            'fallback_predictions': 0,
            'last_mape': float('inf'),
            'last_r2': self.stats.get('last_r2', 0.0) if self.is_ready else 0.0,
        }
        
        if self.config.verbose_logging:
            logger.info("PerformancePredictor reset")
    
    def force_retrain(self) -> bool:
        """强制重新训练模型（即使使用预拟合模型）"""
        if not MODEL_AVAILABLE or len(self.performance_buffer) < self.config.min_samples_for_update:
            return False
        
        # 临时禁用预拟合模式
        original_use_pretrained = self.config.use_pretrained_model
        self.config.use_pretrained_model = False
        
        try:
            result = self._update_model()
            return result
        finally:
            # 恢复原始配置
            self.config.use_pretrained_model = original_use_pretrained

    def get_p_max(self) -> Optional[float]:
        """获取吞吐饱和模型中的 P_max 参数（tokens/ms）。
        若模型未就绪或不可用，返回 None。
        """
        try:
            if self.is_ready and self.model is not None and getattr(self.model, 'P_max', None) is not None:
                return float(self.model.P_max)
        except Exception:
            pass
        return None