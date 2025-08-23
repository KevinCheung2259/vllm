"""
vLLM Scheduler 性能建模包

主要功能:
- 基于吞吐饱和理论的物理启发模型
- 自动数据预处理和特征工程
- 非线性拟合和模型验证
- 可视化分析和性能预测

使用示例:
    from modeling import ThroughputSaturationModel
    
    # 创建模型
    model = ThroughputSaturationModel()
    
    # 拟合数据
    model.fit(profiling_dataframe)
    
    # 预测延迟
    latency = model.predict(batch_size=16, total_tokens=512)
    
    # 生成等高线图
    model.plot_contour()
"""

from .performance_model import ThroughputSaturationModel
from .integration import analyze_with_modeling, compare_models

__version__ = "1.0.0"
__author__ = "vLLM Performance Modeling Team"

__all__ = [
    'ThroughputSaturationModel',
    'analyze_with_modeling', 
    'compare_models'
] 