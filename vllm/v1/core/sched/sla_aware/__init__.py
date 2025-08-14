"""SLA-Aware Scheduling Module for vLLM

This module provides SLA-aware scheduling capabilities based on throughput 
saturation modeling theory. It integrates seamlessly with the existing vLLM 
scheduler while maintaining backward compatibility.

Key Features:
- Throughput saturation-based performance modeling
- Three-phase optimization algorithm
- Adaptive latency target computation
- Minimal code intrusion
- Plug-and-play design
"""

from .config import SLASchedulerConfig
from .performance_predictor import PerformancePredictor
from .sla_scheduler import SLAScheduler

__all__ = ['SLASchedulerConfig', 'PerformancePredictor', 'SLAScheduler']
