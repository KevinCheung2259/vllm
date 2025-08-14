#!/usr/bin/env python3
"""
创建预拟合模型的工具脚本

该脚本用于从历史profiling数据训练一个性能模型，并保存为预拟合模型文件，
供SLA调度器直接使用，避免冷启动问题。
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 添加当前模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from throughput_model import ThroughputSaturationModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_profiling_data(data_path: str) -> pd.DataFrame:
    """从profiling文件加载数据
    
    Args:
        data_path: profiling数据路径（文件或目录）
        
    Returns:
        合并后的DataFrame
    """
    data_path = Path(data_path)
    data = []
    
    if data_path.is_file():
        # 单个文件
        files = [data_path]
    elif data_path.is_dir():
        # 目录中的所有jsonl文件
        files = list(data_path.glob('*.jsonl'))
        if not files:
            raise ValueError(f"No jsonl files found in directory: {data_path}")
    else:
        raise ValueError(f"Path not found: {data_path}")
    
    logger.info(f"Loading data from {len(files)} file(s)...")
    
    # 读取所有文件
    for file_path in sorted(files):
        logger.info(f"Loading: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 跳过前后几行（可能不稳定）
            valid_lines = lines[10:-10] if len(lines) > 20 else lines
            
            for line in valid_lines:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
    
    if not data:
        raise ValueError("No valid data found")
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} records")
    
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """过滤和清理数据
    
    Args:
        df: 原始DataFrame
        
    Returns:
        清理后的DataFrame
    """
    initial_count = len(df)
    
    # 基本过滤
    df = df[df.get('schedule_duration_ms', 0) < 200]
    df = df[df.get('model_run_duration_ms', 0) < 200]
    df = df[df.get('model_run_duration_ms', 0) > 0]
    
    # 需要有valid chunk_sizes
    df = df[df['chunk_sizes'].notna()]
    
    logger.info(f"Data filtering: {initial_count} -> {len(df)} records")
    
    return df


def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """分析数据特征
    
    Args:
        df: 数据DataFrame
        
    Returns:
        数据分析结果
    """
    def compute_batch_size(chunk_sizes):
        if isinstance(chunk_sizes, list):
            return len(chunk_sizes)
        return np.nan
    
    def compute_total_tokens(chunk_sizes):
        if isinstance(chunk_sizes, list) and len(chunk_sizes) > 0:
            return sum(chunk_sizes)
        return np.nan
    
    df['batch_size'] = df['chunk_sizes'].apply(compute_batch_size)
    df['total_tokens'] = df['chunk_sizes'].apply(compute_total_tokens)
    
    # 过滤有效数据
    valid_mask = (
        df['batch_size'].notna() & 
        df['total_tokens'].notna() & 
        (df['batch_size'] > 0) & 
        (df['total_tokens'] > 0)
    )
    df_valid = df[valid_mask]
    
    analysis = {
        'total_records': len(df),
        'valid_records': len(df_valid),
        'batch_size_range': (df_valid['batch_size'].min(), df_valid['batch_size'].max()),
        'total_tokens_range': (df_valid['total_tokens'].min(), df_valid['total_tokens'].max()),
        'latency_range': (df_valid['model_run_duration_ms'].min(), df_valid['model_run_duration_ms'].max()),
        'batch_size_dist': df_valid['batch_size'].value_counts().head(10).to_dict(),
    }
    
    return analysis


def train_model(df: pd.DataFrame, verbose: bool = True) -> ThroughputSaturationModel:
    """训练性能模型
    
    Args:
        df: 训练数据
        verbose: 是否输出详细信息
        
    Returns:
        训练好的模型
    """
    logger.info("Training throughput saturation model...")
    
    # 创建模型
    model = ThroughputSaturationModel(verbose=verbose)
    
    # 训练模型
    model.fit(df)
    
    # 打印训练结果
    summary = model.get_model_summary()
    logger.info(f"Model training completed:")
    logger.info(f"  R² = {summary['metrics']['r2']:.4f}")
    logger.info(f"  RMSE = {summary['metrics']['rmse']:.3f} ms")
    logger.info(f"  MAE = {summary['metrics']['mae']:.3f} ms")
    logger.info(f"  Samples = {summary['metrics']['n_samples']}")
    
    return model


def create_demo_model(output_path: str) -> None:
    """创建演示用的模型（当没有真实数据时）
    
    Args:
        output_path: 输出路径
    """
    logger.info("Creating demo model with synthetic data...")
    
    # 生成合成数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟的batch_sizes和chunk_sizes
    batch_sizes = np.random.randint(1, 33, n_samples)
    chunk_sizes_list = []
    
    for b in batch_sizes:
        if np.random.random() < 0.3:  # 30% prefill
            sizes = np.random.randint(10, 200, b).tolist()
        else:  # 70% decode  
            sizes = [1] * b
        chunk_sizes_list.append(sizes)
    
    # 使用真实的模型参数生成延迟
    true_params = [50.0, 0.1, 0.02, 5.0, 0.05, 10.0, 0.5, 0.001]
    
    latencies = []
    for i, sizes in enumerate(chunk_sizes_list):
        B = len(sizes)
        S = sum(sizes)
        latency = ThroughputSaturationModel.latency_model(
            (np.array([B]), np.array([S])), *true_params
        )[0]
        latency += np.random.normal(0, latency * 0.1)  # 10% 噪声
        latencies.append(max(latency, 1.0))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'chunk_sizes': chunk_sizes_list,
        'model_run_duration_ms': latencies,
        'batch_id': range(n_samples)
    })
    
    # 训练模型
    model = train_model(df, verbose=True)
    
    # 保存模型
    model.save_model(output_path)
    logger.info(f"Demo model saved to: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Create pretrained model for SLA scheduler')
    parser.add_argument('--data', type=str, help='Path to profiling data (file or directory)')
    parser.add_argument('--output', type=str, default='sla_scheduler_pretrained_model.pkl',
                       help='Output model file path')
    parser.add_argument('--demo', action='store_true', 
                       help='Create demo model with synthetic data')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze data without training model')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 SLA调度器预拟合模型创建工具")
    print("=" * 60)
    
    if args.demo:
        # 创建演示模型
        create_demo_model(args.output)
        return
    
    if not args.data:
        print("❌ 错误：需要提供数据路径 (--data) 或使用 --demo 创建演示模型")
        parser.print_help()
        return
    
    try:
        # 加载数据
        df = load_profiling_data(args.data)
        
        # 过滤数据
        df = filter_data(df)
        
        # 分析数据
        analysis = analyze_data(df)
        
        print("\n📊 数据分析结果:")
        print(f"  总记录数: {analysis['total_records']}")
        print(f"  有效记录数: {analysis['valid_records']}")
        print(f"  Batch Size范围: {analysis['batch_size_range']}")
        print(f"  Total Tokens范围: {analysis['total_tokens_range']}")
        print(f"  延迟范围: {analysis['latency_range']} ms")
        
        print(f"\n📈 Batch Size分布:")
        for bs, count in analysis['batch_size_dist'].items():
            print(f"    Batch {bs}: {count} 次")
        
        if args.analyze_only:
            print("\n✅ 数据分析完成（仅分析模式）")
            return
        
        if analysis['valid_records'] < 50:
            print(f"\n⚠️  警告：有效样本数过少 ({analysis['valid_records']})，建议至少50个样本")
            response = input("是否继续训练模型？ (y/N): ")
            if response.lower() != 'y':
                return
        
        # 训练模型
        print(f"\n🔧 开始训练模型...")
        model = train_model(df, verbose=args.verbose)
        
        # 保存模型
        model.save_model(args.output)
        
        print(f"\n✅ 模型训练完成并保存到: {args.output}")
        print(f"\n💡 使用方法:")
        print(f"   export VLLM_SLA_USE_PRETRAINED=true")
        print(f"   export VLLM_SLA_PRETRAINED_PATH='{args.output}'")
        print(f"   # 然后启动vLLM")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
