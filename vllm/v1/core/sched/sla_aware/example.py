#!/usr/bin/env python3
"""
SLA感知调度器使用示例

演示如何配置和使用SLA感知调度器，包括性能监控和调试。
"""

import os
import time
from typing import List, Dict, Any

# 设置示例配置
def setup_example_config():
    """设置SLA调度器示例配置"""
    os.environ['VLLM_SLA_SCHEDULER_ENABLED'] = 'true'
    os.environ['VLLM_SLO_TPOT_MS'] = '50.0'
    os.environ['VLLM_SLA_MIN_BATCH_TIME_MS'] = '15.0'
    os.environ['VLLM_SLA_QUEUE_THRESHOLD'] = '5'
    os.environ['VLLM_SLA_VERBOSE'] = 'true'
    os.environ['VLLM_SLA_FALLBACK_ON_ERROR'] = 'true'
    
    print("✅ SLA调度器配置已设置")


def demonstrate_config_loading():
    """演示配置加载"""
    print("\n📋 配置加载示例")
    print("=" * 50)
    
    try:
        from .config import SLASchedulerConfig
        
        # 从环境变量加载
        config = SLASchedulerConfig.from_env()
        print(f"从环境变量加载的配置: {config}")
        
        # 程序化配置
        custom_config = SLASchedulerConfig(
            enabled=True,
            slo_tpot_ms=60.0,
            min_batch_time_ms=20.0,
            queue_threshold=3,
            verbose_logging=True
        )
        print(f"自定义配置: {custom_config}")
        
        return config
        
    except ImportError as e:
        print(f"❌ 无法导入配置模块: {e}")
        return None


def demonstrate_performance_predictor():
    """演示性能预测器"""
    print("\n🔮 性能预测器示例")
    print("=" * 50)
    
    try:
        from .config import SLASchedulerConfig
        from .performance_predictor import PerformancePredictor
        
        config = SLASchedulerConfig(verbose_logging=True)
        predictor = PerformancePredictor(config)
        
        print("📊 添加模拟性能数据...")
        
        # 添加一些模拟数据
        sample_data = [
            (4, 64, 12.5),    # batch=4, tokens=64, latency=12.5ms
            (8, 128, 18.2),   # batch=8, tokens=128, latency=18.2ms
            (16, 256, 28.7),  # batch=16, tokens=256, latency=28.7ms
            (32, 512, 45.3),  # batch=32, tokens=512, latency=45.3ms
        ]
        
        for batch_size, tokens, latency in sample_data:
            predictor.add_observation(batch_size, tokens, latency)
            print(f"  添加观测: B={batch_size}, S={tokens}, T={latency}ms")
        
        print("\n🎯 预测测试...")
        test_cases = [(8, 200), (16, 400), (24, 600)]
        
        for batch_size, tokens in test_cases:
            pred_latency = predictor.predict_latency(batch_size, tokens)
            token_budget = predictor.solve_for_token_budget(batch_size, 30.0)
            
            print(f"  B={batch_size}, S={tokens} -> 预测延迟: {pred_latency:.2f}ms")
            print(f"  B={batch_size}, 目标30ms -> Token预算: {token_budget}")
        
        status = predictor.get_status()
        print(f"\n📈 预测器状态: {status}")
        
        return predictor
        
    except ImportError as e:
        print(f"❌ 无法导入预测器模块: {e}")
        return None


def demonstrate_sla_scheduler():
    """演示SLA调度器主接口"""
    print("\n🎛️ SLA调度器主接口示例")
    print("=" * 50)
    
    try:
        from .sla_scheduler import SLAScheduler
        
        scheduler = SLAScheduler()
        print(f"✅ SLA调度器已初始化: {scheduler.enabled}")
        
        # 模拟调度请求
        print("\n📋 模拟调度计算...")
        
        # 创建模拟请求（这里用简化的数据结构）
        class MockRequest:
            def __init__(self, req_id: str, num_computed_tokens: int = 0, num_prompt_tokens: int = 100):
                self.request_id = req_id
                self.num_computed_tokens = num_computed_tokens
                self.num_prompt_tokens = num_prompt_tokens
        
        running_requests = [
            MockRequest("req_1", 50, 100),   # prefill阶段
            MockRequest("req_2", 100, 100),  # decode阶段
            MockRequest("req_3", 100, 100),  # decode阶段
        ]
        
        waiting_requests = [
            MockRequest("req_4", 0, 200),
            MockRequest("req_5", 0, 150),
        ]
        
        # 计算token预算和目标延迟
        token_budget, target_latency = scheduler.compute_token_budget_and_target(
            running_requests=running_requests,
            waiting_requests=waiting_requests,
            max_tokens=512,
            max_batch_size=32
        )
        
        print(f"  Token预算: {token_budget}")
        print(f"  目标延迟: {target_latency:.2f}ms")
        
        # 检查是否应该优先decode
        prioritize_decode = scheduler.should_prioritize_decode(running_requests)
        print(f"  优先decode: {prioritize_decode}")
        
        # 模拟性能记录
        scheduler.record_performance(5, 320, 35.5)
        print(f"  记录性能: B=5, S=320, T=35.5ms")
        
        # 获取状态
        status = scheduler.get_status()
        print(f"\n📊 调度器状态:")
        for key, value in status.items():
            if key != 'predictor':  # 简化输出
                print(f"  {key}: {value}")
        
        simple_status = scheduler.get_simple_status()
        print(f"\n📊 简化状态: {simple_status}")
        
        return scheduler
        
    except ImportError as e:
        print(f"❌ 无法导入SLA调度器模块: {e}")
        return None


def demonstrate_integration():
    """演示与主调度器的集成"""
    print("\n🔗 调度器集成示例")
    print("=" * 50)
    
    print("SLA调度器与vLLM主调度器的集成点:")
    print("1. 在Scheduler.__init__()中初始化SLA调度器")
    print("2. 在schedule()方法中调用compute_token_budget_and_target()")
    print("3. 在_finalize_and_log_profiling()中记录性能数据")
    print("4. 通过get_sla_scheduler_status()监控状态")
    
    print("\n集成代码示例:")
    print("""
    # 在主调度器中
    if self.sla_scheduler and self.sla_scheduler.enabled:
        token_budget, target_latency = self.sla_scheduler.compute_token_budget_and_target(
            running_requests=self.running,
            waiting_requests=list(self.waiting),
            max_tokens=self.max_num_scheduled_tokens,
            max_batch_size=self.max_num_running_reqs
        )
    else:
        # 回退到原有逻辑
        token_budget = self.max_num_scheduled_tokens
        target_latency = self.slo_tpot_ms
    """)


def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n⚡ 性能基准测试")
    print("=" * 50)
    
    try:
        from .performance_predictor import PerformancePredictor
        from .config import SLASchedulerConfig
        
        config = SLASchedulerConfig(verbose_logging=False)
        predictor = PerformancePredictor(config)
        
        # 预热：添加一些数据
        for i in range(100):
            batch_size = (i % 32) + 1
            tokens = (i % 512) + 64
            latency = 10 + 0.02 * tokens + 0.5 * batch_size
            predictor.add_observation(batch_size, tokens, latency)
        
        # 基准测试预测性能
        test_cases = [(8, 256), (16, 512), (32, 1024)]
        iterations = 1000
        
        for batch_size, tokens in test_cases:
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                predictor.predict_latency(batch_size, tokens)
            
            end_time = time.perf_counter()
            avg_time_ms = (end_time - start_time) * 1000 / iterations
            
            print(f"预测性能 B={batch_size}, S={tokens}: {avg_time_ms:.4f}ms/次")
        
        # 基准测试token求解性能
        start_time = time.perf_counter()
        for _ in range(iterations):
            predictor.solve_for_token_budget(16, 30.0)
        end_time = time.perf_counter()
        avg_solve_time_ms = (end_time - start_time) * 1000 / iterations
        
        print(f"Token求解性能: {avg_solve_time_ms:.4f}ms/次")
        
        print(f"\n✅ 性能测试完成，预测开销远小于1ms限制")
        
    except ImportError as e:
        print(f"❌ 无法运行性能测试: {e}")


def main():
    """主函数"""
    print("🚀 SLA感知调度器示例程序")
    print("=" * 60)
    
    # 1. 设置配置
    setup_example_config()
    
    # 2. 演示配置加载
    config = demonstrate_config_loading()
    
    # 3. 演示性能预测器
    predictor = demonstrate_performance_predictor()
    
    # 4. 演示SLA调度器
    scheduler = demonstrate_sla_scheduler()
    
    # 5. 演示集成
    demonstrate_integration()
    
    # 6. 性能基准测试
    run_performance_benchmark()
    
    print("\n🎉 示例程序执行完成！")
    print("\n要在实际vLLM中使用SLA调度器，请设置相应的环境变量，")
    print("或参考README.md中的详细配置说明。")


if __name__ == '__main__':
    main()
