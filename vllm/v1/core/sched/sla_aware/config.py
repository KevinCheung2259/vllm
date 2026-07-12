"""SLA调度器配置管理模块

该模块负责管理SLA感知调度器的所有配置参数，支持从环境变量加载配置，
同时提供合理的默认值以确保系统稳定性。
"""

import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SLASchedulerConfig:
    """SLA调度器配置类
    
    包含所有SLA调度器相关的配置参数，支持从环境变量加载，
    并提供与现有vLLM调度器兼容的配置选项。
    """
    
    # === 功能开关 ===
    enabled: bool = False                    # 是否启用SLA调度器
    fallback_on_error: bool = True          # 出错时是否回退到原有调度逻辑
    
    # === 性能模型配置 ===
    model_update_threshold: float = 0.15    # MAPE阈值，超过则更新模型
    min_samples_for_update: int = 50        # 触发模型更新的最少样本数
    max_buffer_size: int = 1000             # 性能数据缓冲区大小
    model_confidence_threshold: float = 0.8 # 模型R²阈值，低于此值使用线性后备
    
    # === 预训练模型配置 ===
    use_stable_cluster_model: bool = False  # 是否使用稳定集群模型
    use_pretrained_model: bool = True      # 是否优先使用预训练模型
    pretrained_model_path: str = "sla_scheduler_model.pkl"         # 预训练模型文件路径（空则使用默认）
    save_trained_model: bool = True         # 是否保存训练后的模型
    model_save_path: str = "sla_scheduler_model.pkl"  # 模型保存路径
    
    # === SLA参数 ===
    slo_tpot_ms: float = 50.0              # TPOT (Time Per Output Token) SLA上限
    slo_ttft_ms: float = 500.0             # TTFT (Time To First Token) SLA上限
    
    # === 负载感知参数 ===
    min_batch_time_ms: float = 15.0        # 最小批处理时间
    queue_threshold: int = 5               # 队列长度阈值，用于自适应延迟计算
    
    # === 优化算法参数 ===
    max_batch_search: int = 64             # 最大batch size搜索范围
    optimization_timeout_ms: float = 1.0   # 优化算法超时时间（毫秒）
    
    # === 线性后备模型参数 ===
    fallback_intercept_ms: float = 8.7     # 后备线性模型截距
    fallback_slope_ms_per_token: float = 0.0215  # 后备线性模型斜率
    
    # === 调试和监控 ===
    verbose_logging: bool = True          # 是否启用详细日志
    performance_logging: bool = True       # 是否记录性能指标
    
    @classmethod
    def from_env(cls) -> 'SLASchedulerConfig':
        """从环境变量加载配置
        
        提供了全面的环境变量支持，使得用户可以在不修改代码的情况下
        调整SLA调度器的所有配置参数。
        
        Returns:
            SLASchedulerConfig: 从环境变量加载的配置实例
        """
        try:
            config = cls(
                # 功能开关
                enabled=os.getenv('VLLM_SLA_SCHEDULER_ENABLED', 'true').lower() == 'true',
                fallback_on_error=os.getenv('VLLM_SLA_FALLBACK_ON_ERROR', 'true').lower() == 'true',
                
                # 性能模型配置
                model_update_threshold=float(os.getenv('VLLM_SLA_MODEL_UPDATE_THRESHOLD', '0.15')),
                min_samples_for_update=int(os.getenv('VLLM_SLA_MIN_SAMPLES', '64')),
                max_buffer_size=int(os.getenv('VLLM_SLA_BUFFER_SIZE', '1000')),
                model_confidence_threshold=float(os.getenv('VLLM_SLA_MODEL_CONFIDENCE', '0.8')),
                
                # SLA参数
                slo_tpot_ms=float(os.getenv('VLLM_SLO_TPOT_MS', '50.0')),
                slo_ttft_ms=float(os.getenv('VLLM_SLO_TTFT_MS', '500.0')),
                
                # 负载感知参数
                min_batch_time_ms=float(os.getenv('VLLM_SLA_MIN_BATCH_TIME_MS', '15.0')),
                queue_threshold=int(os.getenv('VLLM_SLA_QUEUE_THRESHOLD', '5')),
                
                # 优化算法参数
                # max_batch_search=int(os.getenv('VLLM_SLA_MAX_BATCH_SEARCH', '32')),
                optimization_timeout_ms=float(os.getenv('VLLM_SLA_OPT_TIMEOUT_MS', '1.0')),
                
                # 预训练模型配置
                # 默认使用预训练模型时，就不会进行在线训练
                use_stable_cluster_model=os.getenv('VLLM_SLA_USE_STABLE_MODEL', 'true').lower() == 'true',
                use_pretrained_model=os.getenv('VLLM_SLA_USE_PRETRAINED', 'true').lower() == 'true',
                pretrained_model_path=os.getenv('VLLM_SLA_PRETRAINED_PATH', 'stable_model_h100.pkl'),
                save_trained_model=os.getenv('VLLM_SLA_SAVE_MODEL', 'false').lower() == 'true',
                model_save_path=os.getenv('VLLM_SLA_MODEL_SAVE_PATH', 'stable_sla_scheduler_model_v2.pkl'),
                
                # 线性后备模型参数
                fallback_intercept_ms=float(os.getenv('VLLM_SLA_FALLBACK_INTERCEPT', '8.7')),
                fallback_slope_ms_per_token=float(os.getenv('VLLM_SLA_FALLBACK_SLOPE', '0.0215')),
                
                # 调试和监控
                verbose_logging=os.getenv('VLLM_SLA_VERBOSE', 'false').lower() == 'true',
                performance_logging=os.getenv('VLLM_SLA_PERF_LOG', 'false').lower() == 'true',
            )
            
            # 验证配置合理性
            config._validate_config()
            
            if config.verbose_logging:
                logger.info(f"SLA Scheduler config loaded from environment: {config}")
                
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load SLA config from environment: {e}, using defaults")
            return cls()  # 使用默认配置
    
    def _validate_config(self) -> None:
        """验证配置参数的合理性"""
        if self.slo_tpot_ms <= 0:
            raise ValueError("slo_tpot_ms must be positive")
        
        if self.min_batch_time_ms <= 0:
            raise ValueError("min_batch_time_ms must be positive")
            
        if self.min_batch_time_ms >= self.slo_tpot_ms:
            logger.warning(f"min_batch_time_ms ({self.min_batch_time_ms}) >= slo_tpot_ms ({self.slo_tpot_ms})")
        
        if self.model_update_threshold <= 0 or self.model_update_threshold >= 1:
            raise ValueError("model_update_threshold must be in (0, 1)")
            
        if self.max_batch_search <= 0:
            raise ValueError("max_batch_search must be positive")
            
        if self.optimization_timeout_ms <= 0:
            raise ValueError("optimization_timeout_ms must be positive")
            
    def to_dict(self) -> dict:
        """转换为字典，便于日志记录和序列化"""
        return {
            'enabled': self.enabled,
            'fallback_on_error': self.fallback_on_error,
            'slo_tpot_ms': self.slo_tpot_ms,
            'min_batch_time_ms': self.min_batch_time_ms,
            'queue_threshold': self.queue_threshold,
            'max_batch_search': self.max_batch_search,
            'model_update_threshold': self.model_update_threshold,
            'min_samples_for_update': self.min_samples_for_update,
            'use_pretrained_model': self.use_pretrained_model,
            'pretrained_model_path': self.pretrained_model_path,
            'save_trained_model': self.save_trained_model,
            'use_stable_cluster_model': self.use_stable_cluster_model,
        }
    
    def __str__(self) -> str:
        """字符串表示，用于日志输出"""
        pretrained_info = f", pretrained={self.use_pretrained_model}"
        if self.use_pretrained_model and self.pretrained_model_path:
            pretrained_info += f"({self.pretrained_model_path})"
            
        stable_info = ""
        if self.use_stable_cluster_model:
            stable_info = f", stable_model={self.use_stable_cluster_model}"
            
        return f"SLAConfig(enabled={self.enabled}, slo_tpot={self.slo_tpot_ms}ms, min_batch={self.min_batch_time_ms}ms{pretrained_info}{stable_info})"
