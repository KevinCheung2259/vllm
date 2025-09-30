#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析scheduler profiling数据中model_run_duration_ms的变化原因
筛选total_scheduled_tokens=4096的记录进行深入分析
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_and_filter_data(file_path):
    """加载并筛选total_scheduled_tokens=4096的数据"""
    data = []
    
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                if record.get('total_scheduled_tokens') == 4096:
                    data.append(record)
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                continue
    
    print(f"找到 {len(data)} 条total_scheduled_tokens=4096的记录")
    return pd.DataFrame(data)

def add_derived_features(df):
    """添加派生特征"""
    # 计算总的已计算token数
    df['total_computed_tokens'] = df['all_computed_tokens'].apply(sum)
    
    # 计算总的缓存token数
    df['total_cached_tokens'] = df['all_cached_tokens'].apply(sum)
    
    # 计算chunk数量
    df['num_chunks'] = df['chunk_sizes'].apply(len)
    
    # 计算最大chunk大小
    df['max_chunk_size'] = df['chunk_sizes'].apply(max)
    
    # 计算最小chunk大小
    df['min_chunk_size'] = df['chunk_sizes'].apply(min)
    
    # 计算chunk大小的标准差（衡量chunk大小的分散程度）
    df['chunk_size_std'] = df['chunk_sizes'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    
    # 计算平均每个chunk的computed tokens
    df['avg_computed_per_chunk'] = df.apply(lambda row: np.mean(row['all_computed_tokens']) if row['all_computed_tokens'] else 0, axis=1)
    
    # 计算平均每个chunk的cached tokens
    df['avg_cached_per_chunk'] = df.apply(lambda row: np.mean(row['all_cached_tokens']) if row['all_cached_tokens'] else 0, axis=1)
    
    # 计算缓存命中率
    df['cache_hit_ratio'] = df['total_cached_tokens'] / (df['total_computed_tokens'] + df['total_cached_tokens'] + 1e-8)
    
    # 计算是否有大的computed tokens（可能表示prefill阶段）
    df['has_large_computed'] = df['all_computed_tokens'].apply(lambda x: any(token > 1000 for token in x))
    
    # 计算computed tokens的最大值
    df['max_computed_tokens'] = df['all_computed_tokens'].apply(lambda x: max(x) if x else 0)
    
    return df

def analyze_correlations(df):
    """分析相关性"""
    print("\n=== 相关性分析 ===")
    
    # 选择数值型特征进行相关性分析
    numeric_features = [
        'model_run_duration_ms', 'schedule_duration_ms', 'num_waiting_reqs', 
        'num_running_reqs', 'total_computed_tokens', 'total_cached_tokens',
        'num_chunks', 'max_chunk_size', 'min_chunk_size', 'chunk_size_std',
        'avg_computed_per_chunk', 'avg_cached_per_chunk', 'cache_hit_ratio',
        'max_computed_tokens'
    ]
    
    correlation_matrix = df[numeric_features].corr()
    
    # 打印与model_run_duration_ms相关性最强的特征
    model_duration_corr = correlation_matrix['model_run_duration_ms'].abs().sort_values(ascending=False)
    print("与model_run_duration_ms相关性排序:")
    for feature, corr in model_duration_corr.items():
        if feature != 'model_run_duration_ms':
            print(f"  {feature}: {corr:.4f}")
    
    return correlation_matrix

def create_visualizations(df, correlation_matrix):
    """创建可视化图表"""
    print("\n=== Creating Visualizations ===")
    
    # 创建图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. model_run_duration_ms的分布
    plt.subplot(3, 4, 1)
    plt.hist(df['model_run_duration_ms'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Model Run Duration Distribution')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    
    # 2. 相关性热力图
    plt.subplot(3, 4, 2)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 3. model_run_duration_ms vs num_chunks
    plt.subplot(3, 4, 3)
    plt.scatter(df['num_chunks'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Number of Chunks')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Number of Chunks')
    
    # 4. model_run_duration_ms vs max_computed_tokens
    plt.subplot(3, 4, 4)
    plt.scatter(df['max_computed_tokens'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Max Computed Tokens')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Max Computed Tokens')
    
    # 5. model_run_duration_ms vs cache_hit_ratio
    plt.subplot(3, 4, 5)
    plt.scatter(df['cache_hit_ratio'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Cache Hit Ratio')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Cache Hit Ratio')
    
    # 6. model_run_duration_ms vs num_running_reqs
    plt.subplot(3, 4, 6)
    plt.scatter(df['num_running_reqs'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Number of Running Requests')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Number of Running Requests')
    
    # 7. 按是否有大computed tokens分组的duration分布
    plt.subplot(3, 4, 7)
    has_large = df[df['has_large_computed']]['model_run_duration_ms']
    no_large = df[~df['has_large_computed']]['model_run_duration_ms']
    plt.hist([has_large, no_large], bins=30, alpha=0.7, label=['With Large Computed', 'Without Large Computed'])
    plt.xlabel('Model Run Duration (ms)')
    plt.ylabel('Frequency')
    plt.title('Duration Distribution by Computed Tokens Size')
    plt.legend()
    
    # 8. chunk_size_std vs model_run_duration_ms
    plt.subplot(3, 4, 8)
    plt.scatter(df['chunk_size_std'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Chunk Size Std Dev')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Chunk Size Variation')
    
    # 9. model_run_duration_ms vs total_computed_tokens
    plt.subplot(3, 4, 9)
    plt.scatter(df['total_computed_tokens'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Total Computed Tokens')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Total Computed Tokens')
    
    # 10. 按num_chunks分组的boxplot
    plt.subplot(3, 4, 10)
    df_plot = df.copy()
    df_plot['chunk_group'] = pd.cut(df_plot['num_chunks'], bins=5, labels=False)
    sns.boxplot(data=df_plot, x='chunk_group', y='model_run_duration_ms')
    plt.xlabel('Chunk Group')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration Distribution by Number of Chunks')
    
    # 11. schedule_duration_ms vs model_run_duration_ms
    plt.subplot(3, 4, 11)
    plt.scatter(df['schedule_duration_ms'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Schedule Duration (ms)')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Model Duration vs Schedule Duration')
    
    # 12. max_chunk_size vs model_run_duration_ms
    plt.subplot(3, 4, 12)
    plt.scatter(df['max_chunk_size'], df['model_run_duration_ms'], alpha=0.6)
    plt.xlabel('Max Chunk Size')
    plt.ylabel('Model Run Duration (ms)')
    plt.title('Duration vs Max Chunk Size')
    
    plt.tight_layout()
    plt.savefig('model_run_duration_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: model_run_duration_analysis.png")

def detailed_analysis(df):
    """详细分析"""
    print("\n=== 详细分析结果 ===")
    
    # 基本统计信息
    print(f"数据概览:")
    print(f"  记录总数: {len(df)}")
    print(f"  Model Run Duration 统计:")
    print(f"    平均值: {df['model_run_duration_ms'].mean():.2f} ms")
    print(f"    标准差: {df['model_run_duration_ms'].std():.2f} ms")
    print(f"    最小值: {df['model_run_duration_ms'].min():.2f} ms")
    print(f"    最大值: {df['model_run_duration_ms'].max():.2f} ms")
    print(f"    变异系数: {df['model_run_duration_ms'].std() / df['model_run_duration_ms'].mean():.4f}")
    
    # 分析影响因素
    print(f"\n主要影响因素分析:")
    
    # 1. Chunk数量的影响
    chunk_groups = df.groupby('num_chunks')['model_run_duration_ms'].agg(['mean', 'std', 'count'])
    print(f"\n1. Chunk数量对Duration的影响:")
    for chunks, stats in chunk_groups.iterrows():
        print(f"   {chunks}个chunks: 平均{stats['mean']:.2f}ms (std={stats['std']:.2f}, n={stats['count']})")
    
    # 2. 大Computed Tokens的影响
    large_computed_stats = df.groupby('has_large_computed')['model_run_duration_ms'].agg(['mean', 'std', 'count'])
    print(f"\n2. 大Computed Tokens的影响:")
    for has_large, stats in large_computed_stats.iterrows():
        label = "有大Computed" if has_large else "无大Computed"
        print(f"   {label}: 平均{stats['mean']:.2f}ms (std={stats['std']:.2f}, n={stats['count']})")
    
    # 3. 缓存命中率的影响
    df['cache_hit_group'] = pd.cut(df['cache_hit_ratio'], bins=5, labels=['很低', '低', '中', '高', '很高'])
    cache_stats = df.groupby('cache_hit_group')['model_run_duration_ms'].agg(['mean', 'std', 'count'])
    print(f"\n3. 缓存命中率对Duration的影响:")
    for group, stats in cache_stats.iterrows():
        print(f"   {group}缓存命中率: 平均{stats['mean']:.2f}ms (std={stats['std']:.2f}, n={stats['count']})")
    
    # 4. 运行请求数的影响
    running_stats = df.groupby('num_running_reqs')['model_run_duration_ms'].agg(['mean', 'std', 'count'])
    print(f"\n4. 运行请求数对Duration的影响:")
    for num_reqs, stats in running_stats.iterrows():
        print(f"   {num_reqs}个请求: 平均{stats['mean']:.2f}ms (std={stats['std']:.2f}, n={stats['count']})")
    
    # 5. 异常值分析
    print(f"\n5. 异常值分析:")
    q1 = df['model_run_duration_ms'].quantile(0.25)
    q3 = df['model_run_duration_ms'].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold_high = q3 + 1.5 * iqr
    outlier_threshold_low = q1 - 1.5 * iqr
    
    outliers = df[(df['model_run_duration_ms'] > outlier_threshold_high) | 
                  (df['model_run_duration_ms'] < outlier_threshold_low)]
    
    print(f"   异常值阈值: [{outlier_threshold_low:.2f}, {outlier_threshold_high:.2f}] ms")
    print(f"   发现 {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.1f}%)")
    
    if len(outliers) > 0:
        print(f"   异常值特征:")
        print(f"     平均Chunk数量: {outliers['num_chunks'].mean():.1f}")
        print(f"     平均最大Computed Tokens: {outliers['max_computed_tokens'].mean():.0f}")
        print(f"     平均缓存命中率: {outliers['cache_hit_ratio'].mean():.3f}")
        print(f"     平均运行请求数: {outliers['num_running_reqs'].mean():.1f}")

def main():
    """主函数"""
    file_path = '/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result_true/scheduler_profiling_chunk_4096.jsonl'
    
    if not Path(file_path).exists():
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    # 加载和筛选数据
    df = load_and_filter_data(file_path)
    
    if df.empty:
        print("没有找到total_scheduled_tokens=4096的记录")
        return
    
    # 添加派生特征
    df = add_derived_features(df)
    
    # 分析相关性
    correlation_matrix = analyze_correlations(df)
    
    # 创建可视化
    create_visualizations(df, correlation_matrix)
    
    # 详细分析
    detailed_analysis(df)
    
    # 保存处理后的数据
    output_file = '/home/paperspace/zhangy/vllm-workspace/vllm/exp/filtered_data_4096.csv'
    df.to_csv(output_file, index=False)
    print(f"\n筛选后的数据已保存到: {output_file}")
    
    print(f"\n=== 总结 ===")
    print(f"通过分析total_scheduled_tokens=4096的记录，发现影响model_run_duration_ms的主要因素包括:")
    print(f"1. Chunk数量和大小分布")
    print(f"2. Computed Tokens的大小（特别是是否包含大的prefill）")
    print(f"3. 缓存命中率")
    print(f"4. 并发运行的请求数量")
    print(f"5. Chunk大小的变异程度")

if __name__ == "__main__":
    main() 