#!/usr/bin/env python3
"""
vLLM Scheduler 集成模块

将反向优化算法与现有性能建模系统集成，提供完整的调度决策工具。
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# 导入现有模块
from performance_model import ThroughputSaturationModel
from inverse_scheduler import InverseScheduler, RequestInfo, RequestStage, BatchConfig

logger = logging.getLogger(__name__)


class SchedulerOptimizer:
    """完整的调度器优化工具"""
    
    def __init__(self, fitted_model_path: Optional[str] = None, verbose: bool = True):
        """
        初始化调度优化器
        
        Args:
            fitted_model_path: 已训练模型路径
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.model = None
        self.inverse_scheduler = None
        
        if fitted_model_path and Path(fitted_model_path).exists():
            self.load_fitted_model(fitted_model_path)
        
    def load_fitted_model(self, model_path: str):
        """加载已训练的性能模型"""
        try:
            self.model = ThroughputSaturationModel(verbose=self.verbose)
            self.model.load_model(model_path)
            
            # 提取模型参数用于反向调度
            params = dict(zip(self.model.param_names, self.model.params))
            self.inverse_scheduler = InverseScheduler(params, verbose=self.verbose)
            
            if self.verbose:
                logger.info(f"✅ 模型加载成功: {model_path}")
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def fit_model_from_data(self, profiling_data_path: str):
        """从profiling数据训练新模型"""
        from integration import read_profiling_data
        
        # 读取数据
        df = read_profiling_data(profiling_data_path)
        if df is None or df.empty:
            raise ValueError("无法读取有效的profiling数据")
        
        # 训练模型
        self.model = ThroughputSaturationModel(verbose=self.verbose)
        self.model.fit(df)
        
        # 创建反向调度器
        params = dict(zip(self.model.param_names, self.model.params))
        self.inverse_scheduler = InverseScheduler(params, verbose=self.verbose)
        
        if self.verbose:
            logger.info("✅ 模型训练完成")
    
    def predict_latency(self, batch_size: int, total_tokens: int) -> float:
        """预测给定配置的延迟"""
        if not self.model:
            raise ValueError("模型尚未初始化")
        return self.model.predict(batch_size, total_tokens)
    
    def optimize_for_latency(self, requests: List[RequestInfo], 
                           target_latency: float) -> Optional[BatchConfig]:
        """为目标延迟优化批次配置"""
        if not self.inverse_scheduler:
            raise ValueError("反向调度器尚未初始化")
        return self.inverse_scheduler.optimize_batch(requests, target_latency)
    
    def optimize_with_constraints(self, requests: List[RequestInfo],
                                target_latency: float,
                                max_chunk_size: int = 512,
                                min_prefill_ratio: float = 0.0) -> Optional[BatchConfig]:
        """带约束的批次优化"""
        if not self.inverse_scheduler:
            raise ValueError("反向调度器尚未初始化")
        return self.inverse_scheduler.optimize_with_constraints(
            requests, target_latency, max_chunk_size, min_prefill_ratio
        )
    
    def analyze_latency_sensitivity(self, base_requests: List[RequestInfo],
                                  latency_range: Tuple[float, float],
                                  n_points: int = 20) -> Dict:
        """分析延迟敏感性"""
        if not self.inverse_scheduler:
            raise ValueError("反向调度器尚未初始化")
        
        latencies = np.linspace(latency_range[0], latency_range[1], n_points)
        results = {
            'target_latencies': latencies,
            'achieved_latencies': [],
            'total_tokens': [],
            'decode_counts': [],
            'prefill_counts': [],
            'avg_prefill_sizes': [],
            'feasible': []
        }
        
        for target_lat in latencies:
            config = self.inverse_scheduler.optimize_batch(base_requests, target_lat)
            
            if config:
                results['achieved_latencies'].append(config.predicted_latency)
                results['total_tokens'].append(config.total_tokens)
                
                decode_count = sum(1 for cs in config.chunk_sizes if cs == 1)
                prefill_count = len(config.chunk_sizes) - decode_count
                avg_prefill = np.mean([cs for cs in config.chunk_sizes if cs > 1]) if prefill_count > 0 else 0
                
                results['decode_counts'].append(decode_count)
                results['prefill_counts'].append(prefill_count)
                results['avg_prefill_sizes'].append(avg_prefill)
                results['feasible'].append(True)
            else:
                # 填充空值
                results['achieved_latencies'].append(np.nan)
                results['total_tokens'].append(np.nan)
                results['decode_counts'].append(np.nan)
                results['prefill_counts'].append(np.nan)
                results['avg_prefill_sizes'].append(np.nan)
                results['feasible'].append(False)
        
        return results
    
    def plot_optimization_surface(self, request_counts: List[int],
                                latency_targets: List[float],
                                prefill_ratio: float = 0.3,
                                save_path: str = './modeling/optimization_surface.png'):
        """绘制优化结果的3D表面"""
        if not self.inverse_scheduler:
            raise ValueError("反向调度器尚未初始化")
        
        # 创建网格
        R_mesh, L_mesh = np.meshgrid(request_counts, latency_targets)
        T_mesh = np.full_like(R_mesh, np.nan)
        
        # 计算每个点的最优token数
        for i, n_req in enumerate(request_counts):
            for j, target_lat in enumerate(latency_targets):
                # 生成示例请求
                requests = []
                for k in range(n_req):
                    if np.random.random() < prefill_ratio:
                        stage = RequestStage.PREFILL
                        remaining = np.random.randint(50, 300)
                    else:
                        stage = RequestStage.DECODE
                        remaining = 1
                    
                    requests.append(RequestInfo(
                        request_id=f"req_{k}",
                        stage=stage,
                        remaining_tokens=remaining
                    ))
                
                # 优化
                config = self.inverse_scheduler.optimize_batch(requests, target_lat)
                if config:
                    T_mesh[j, i] = config.total_tokens
        
        # 绘制3D表面
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 过滤有效数据点
        mask = ~np.isnan(T_mesh)
        if mask.any():
            surf = ax.plot_surface(R_mesh, L_mesh, T_mesh, 
                                 cmap='viridis', alpha=0.8,
                                 linewidth=0, antialiased=True)
            
            ax.set_xlabel('Number of Requests')
            ax.set_ylabel('Target Latency (ms)')
            ax.set_zlabel('Optimal Total Tokens')
            ax.set_title(f'Scheduler Optimization Surface\n(Prefill Ratio: {prefill_ratio*100:.0f}%)')
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if self.verbose:
            logger.info(f"📊 Optimization surface plot saved: {save_path}")
        
        return fig
    
    def benchmark_scheduler_performance(self, test_scenarios: List[Dict]) -> pd.DataFrame:
        """基准测试调度器性能"""
        if not self.inverse_scheduler:
            raise ValueError("反向调度器尚未初始化")
        
        results = []
        
        for scenario in test_scenarios:
            n_requests = scenario['n_requests']
            target_latency = scenario['target_latency']
            prefill_ratio = scenario.get('prefill_ratio', 0.3)
            
            # 生成测试请求
            requests = []
            for i in range(n_requests):
                if np.random.random() < prefill_ratio:
                    stage = RequestStage.PREFILL
                    remaining = np.random.randint(50, 500)
                else:
                    stage = RequestStage.DECODE
                    remaining = 1
                
                requests.append(RequestInfo(
                    request_id=f"req_{i}",
                    stage=stage,
                    remaining_tokens=remaining,
                    priority=scenario.get('priority', 1.0)
                ))
            
            # 优化
            import time
            start_time = time.time()
            config = self.inverse_scheduler.optimize_batch(requests, target_latency)
            optimization_time = time.time() - start_time
            
            # 记录结果
            if config:
                error = abs(config.predicted_latency - target_latency)
                relative_error = error / target_latency * 100
                
                results.append({
                    'n_requests': n_requests,
                    'target_latency': target_latency,
                    'prefill_ratio': prefill_ratio,
                    'achieved_latency': config.predicted_latency,
                    'total_tokens': config.total_tokens,
                    'error_ms': error,
                    'error_percent': relative_error,
                    'optimization_time_ms': optimization_time * 1000,
                    'feasible': True
                })
            else:
                results.append({
                    'n_requests': n_requests,
                    'target_latency': target_latency,
                    'prefill_ratio': prefill_ratio,
                    'achieved_latency': np.nan,
                    'total_tokens': np.nan,
                    'error_ms': np.nan,
                    'error_percent': np.nan,
                    'optimization_time_ms': optimization_time * 1000,
                    'feasible': False
                })
        
        return pd.DataFrame(results)
    
    def compare_scheduling_strategies(self, requests: List[RequestInfo],
                                    target_latency: float) -> Dict:
        """比较不同调度策略"""
        if not self.model or not self.inverse_scheduler:
            raise ValueError("模型尚未初始化")
        
        results = {}
        
        # 1. 反向优化策略
        config_optimal = self.inverse_scheduler.optimize_batch(requests, target_latency)
        if config_optimal:
            results['optimal'] = {
                'total_tokens': config_optimal.total_tokens,
                'predicted_latency': config_optimal.predicted_latency,
                'chunk_sizes': config_optimal.chunk_sizes,
                'strategy': 'Inverse Optimization'
            }
        
        # 2. 均匀分配策略
        n_decode = sum(1 for r in requests if r.stage == RequestStage.DECODE)
        n_prefill = len(requests) - n_decode
        
        if config_optimal:
            # 基于最优总token数均匀分配
            uniform_prefill_size = max(1, (config_optimal.total_tokens - n_decode) // max(n_prefill, 1))
            uniform_chunk_sizes = []
            for req in requests:
                if req.stage == RequestStage.DECODE:
                    uniform_chunk_sizes.append(1)
                else:
                    uniform_chunk_sizes.append(uniform_prefill_size)
            
            uniform_total = sum(uniform_chunk_sizes)
            uniform_latency = self.model.predict(len(requests), uniform_total)
            
            results['uniform'] = {
                'total_tokens': uniform_total,
                'predicted_latency': uniform_latency,
                'chunk_sizes': uniform_chunk_sizes,
                'strategy': 'Uniform Allocation'
            }
        
        # 3. 贪心策略（简化版）
        greedy_chunk_sizes = []
        for req in requests:
            if req.stage == RequestStage.DECODE:
                greedy_chunk_sizes.append(1)
            else:
                # 简单贪心：基于剩余token需求
                greedy_chunk_sizes.append(min(req.remaining_tokens, 128))
        
        greedy_total = sum(greedy_chunk_sizes)
        greedy_latency = self.model.predict(len(requests), greedy_total)
        
        results['greedy'] = {
            'total_tokens': greedy_total,
            'predicted_latency': greedy_latency,
            'chunk_sizes': greedy_chunk_sizes,
            'strategy': 'Greedy Allocation'
        }
        
        return results


def demo_scheduler_integration():
    """演示完整的调度器集成功能"""
    print("🚀 vLLM 调度器集成演示")
    print("=" * 50)
    
    # 方式1：使用模拟数据训练新模型
    print("\n📊 步骤1: 训练性能模型")
    print("-" * 30)
    
    # 生成模拟训练数据
    np.random.seed(42)
    training_data = []
    for i in range(1000):
        B = np.random.randint(1, 33)
        if np.random.random() < 0.7:
            chunk_sizes = [1] * B  # decode-only
        else:
            chunk_sizes = []
            for _ in range(B):
                if np.random.random() < 0.3:
                    chunk_sizes.append(np.random.randint(10, 200))
                else:
                    chunk_sizes.append(1)
        
        S = sum(chunk_sizes)
        # 使用真实模型生成延迟
        latency = 10 + 0.02 * S + 0.1 * B + np.random.normal(0, 2)
        latency = max(latency, 1.0)
        
        training_data.append({
            'chunk_sizes': chunk_sizes,
            'model_run_duration_ms': latency,
            'batch_id': i
        })
    
    df_train = pd.DataFrame(training_data)
    
    # 创建优化器并训练
    optimizer = SchedulerOptimizer(verbose=True)
    
    # 手动训练模型（避免文件依赖）
    optimizer.model = ThroughputSaturationModel(verbose=False)
    optimizer.model.fit(df_train)
    
    params = dict(zip(optimizer.model.param_names, optimizer.model.params))
    optimizer.inverse_scheduler = InverseScheduler(params, verbose=False)
    
    print("✅ 模型训练完成")
    
    # 步骤2：反向优化演示
    print("\n🎯 步骤2: 反向调度优化")
    print("-" * 30)
    
    # 创建测试请求
    test_requests = [
        RequestInfo("req_0", RequestStage.DECODE),
        RequestInfo("req_1", RequestStage.DECODE),
        RequestInfo("req_2", RequestStage.PREFILL, remaining_tokens=100),
        RequestInfo("req_3", RequestStage.PREFILL, remaining_tokens=200),
        RequestInfo("req_4", RequestStage.DECODE),
    ]
    
    target_latency = 25.0
    config = optimizer.optimize_for_latency(test_requests, target_latency)
    
    if config:
        print(f"✅ 优化成功!")
        print(f"   目标延迟: {target_latency:.1f}ms")
        print(f"   实际延迟: {config.predicted_latency:.1f}ms")
        print(f"   误差: {abs(config.predicted_latency - target_latency):.1f}ms")
        print(f"   总token数: {config.total_tokens}")
        print(f"   chunk分配: {config.chunk_sizes}")
    
    # 步骤3：策略比较
    print("\n📈 步骤3: 调度策略比较")
    print("-" * 30)
    
    strategies = optimizer.compare_scheduling_strategies(test_requests, target_latency)
    
    print(f"{'策略':<20} {'延迟(ms)':<12} {'Token数':<10} {'误差(ms)':<10}")
    print("-" * 55)
    
    for name, result in strategies.items():
        error = abs(result['predicted_latency'] - target_latency)
        print(f"{result['strategy']:<20} {result['predicted_latency']:<12.1f} {result['total_tokens']:<10} {error:<10.1f}")
    
    # 步骤4：基准测试
    print("\n⚡ 步骤4: 性能基准测试")
    print("-" * 30)
    
    test_scenarios = [
        {'n_requests': 8, 'target_latency': 20.0, 'prefill_ratio': 0.2},
        {'n_requests': 16, 'target_latency': 30.0, 'prefill_ratio': 0.3},
        {'n_requests': 32, 'target_latency': 50.0, 'prefill_ratio': 0.4},
    ]
    
    benchmark_results = optimizer.benchmark_scheduler_performance(test_scenarios)
    
    print("基准测试结果:")
    for _, row in benchmark_results.iterrows():
        if row['feasible']:
            print(f"  请求数={row['n_requests']:2d}, 目标={row['target_latency']:4.1f}ms, "
                  f"实际={row['achieved_latency']:4.1f}ms, 误差={row['error_percent']:4.1f}%, "
                  f"优化时间={row['optimization_time_ms']:5.2f}ms")
        else:
            print(f"  请求数={row['n_requests']:2d}, 目标={row['target_latency']:4.1f}ms, 不可行")
    
    print("\n🎉 演示完成！")


if __name__ == '__main__':
    demo_scheduler_integration() 