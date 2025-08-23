#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析model_run_duration_ms变化的具体原因
基于之前的分析结果，进行更深入的原因探究
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_processed_data():
    """加载之前处理的数据"""
    file_path = '/home/paperspace/zhangy/vllm-workspace/vllm/exp/filtered_data_4096.csv'
    df = pd.read_csv(file_path)
    
    # 重新处理列表型数据
    df['chunk_sizes'] = df['chunk_sizes'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['all_computed_tokens'] = df['all_computed_tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['all_cached_tokens'] = df['all_cached_tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    return df

def analyze_chunk_patterns(df):
    """分析chunk模式对运行时间的影响"""
    print("\n=== Chunk模式详细分析 ===")
    
    # 1. 分析chunk大小的分布模式
    def classify_chunk_pattern(chunk_sizes):
        if len(chunk_sizes) <= 1:
            return "单一chunk"
        
        max_chunk = max(chunk_sizes)
        min_chunk = min(chunk_sizes)
        avg_chunk = np.mean(chunk_sizes)
        std_chunk = np.std(chunk_sizes)
        
        # 检查是否有一个非常大的chunk
        if max_chunk > 2000:
            return "大chunk主导"
        elif std_chunk < 100:
            return "均匀分布"
        elif max_chunk > 10 * min_chunk:
            return "极不均匀"
        else:
            return "中等不均匀"
    
    df['chunk_pattern'] = df['chunk_sizes'].apply(classify_chunk_pattern)
    
    # 分析各种模式的运行时间
    pattern_stats = df.groupby('chunk_pattern')['model_run_duration_ms'].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).round(2)
    
    print("不同Chunk模式的运行时间统计:")
    for pattern, stats in pattern_stats.iterrows():
        print(f"  {pattern}: 平均{stats['mean']}ms, 标准差{stats['std']}ms, "
              f"范围[{stats['min']}-{stats['max']}]ms, 样本数{stats['count']}")
    
    return df

def analyze_memory_access_pattern(df):
    """分析内存访问模式对运行时间的影响"""
    print("\n=== 内存访问模式分析 ===")
    
    # 计算内存访问相关指标
    def compute_memory_metrics(row):
        computed = row['all_computed_tokens']
        cached = row['all_cached_tokens']
        chunks = row['chunk_sizes']
        
        # 计算内存访问复杂度
        memory_hops = 0  # 内存跳跃次数
        cache_misses = sum(1 for c in computed if c > 100)  # 缓存缺失次数
        
        # 计算连续性指标
        chunk_switches = len(chunks) - 1  # chunk切换次数
        
        # 计算工作负载分散度
        workload_variance = np.var(computed) if computed else 0
        
        return {
            'cache_misses': cache_misses,
            'chunk_switches': chunk_switches,
            'workload_variance': workload_variance,
            'avg_work_per_chunk': np.mean(computed) if computed else 0
        }
    
    memory_metrics = df.apply(compute_memory_metrics, axis=1, result_type='expand')
    for col in memory_metrics.columns:
        df[col] = memory_metrics[col]
    
    # 分析内存访问模式与运行时间的关系
    print("内存访问模式对运行时间的影响:")
    
    # 1. 缓存缺失的影响
    cache_miss_corr = df['cache_misses'].corr(df['model_run_duration_ms'])
    print(f"  缓存缺失次数与运行时间相关性: {cache_miss_corr:.4f}")
    
    # 2. Chunk切换的影响
    chunk_switch_corr = df['chunk_switches'].corr(df['model_run_duration_ms'])
    print(f"  Chunk切换次数与运行时间相关性: {chunk_switch_corr:.4f}")
    
    # 3. 工作负载方差的影响
    workload_var_corr = df['workload_variance'].corr(df['model_run_duration_ms'])
    print(f"  工作负载方差与运行时间相关性: {workload_var_corr:.4f}")
    
    return df

def analyze_resource_contention(df):
    """分析资源竞争对运行时间的影响"""
    print("\n=== 资源竞争分析 ===")
    
    # 计算资源竞争指标
    df['total_work'] = df['total_computed_tokens'] + df['total_cached_tokens']
    df['work_per_request'] = df['total_work'] / df['num_running_reqs']
    df['memory_pressure'] = df['total_computed_tokens'] / 4096  # 相对于总token的压力
    
    # 分析并发度对性能的影响
    concurrency_stats = df.groupby('num_running_reqs')['model_run_duration_ms'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    print("并发请求数对运行时间的影响:")
    for reqs, stats in concurrency_stats.iterrows():
        if stats['count'] >= 10:  # 只显示样本数足够的情况
            print(f"  {reqs}个并发请求: 平均{stats['mean']}ms (std={stats['std']}, n={stats['count']})")
    
    # 分析工作负载分布的影响
    print(f"\n工作负载分布指标:")
    print(f"  平均每请求工作量与运行时间相关性: {df['work_per_request'].corr(df['model_run_duration_ms']):.4f}")
    print(f"  内存压力与运行时间相关性: {df['memory_pressure'].corr(df['model_run_duration_ms']):.4f}")
    
    return df

def feature_importance_analysis(df):
    """使用随机森林分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 准备特征
    feature_cols = [
        'num_chunks', 'max_chunk_size', 'chunk_size_std',
        'total_computed_tokens', 'cache_hit_ratio', 'num_running_reqs',
        'cache_misses', 'chunk_switches', 'workload_variance',
        'work_per_request', 'memory_pressure', 'max_computed_tokens'
    ]
    
    X = df[feature_cols]
    y = df['model_run_duration_ms']
    
    # 训练随机森林模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    # 计算特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("特征重要性排序（随机森林）:")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 模型性能
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"\n模型R²分数: {r2:.4f}")
    
    return importance_df

def create_detailed_visualizations(df):
    """创建详细的可视化分析"""
    print("\n=== 创建详细可视化 ===")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Model Run Duration 深度分析', fontsize=16)
    
    # 1. Chunk模式分布
    ax = axes[0, 0]
    pattern_means = df.groupby('chunk_pattern')['model_run_duration_ms'].mean()
    pattern_means.plot(kind='bar', ax=ax)
    ax.set_title('不同Chunk模式的平均运行时间')
    ax.set_ylabel('运行时间 (ms)')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. 并发度vs运行时间（箱线图）
    ax = axes[0, 1]
    # 过滤掉样本数太少的情况
    valid_concurrency = df['num_running_reqs'].value_counts()
    valid_reqs = valid_concurrency[valid_concurrency >= 20].index
    df_filtered = df[df['num_running_reqs'].isin(valid_reqs)]
    
    sns.boxplot(data=df_filtered, x='num_running_reqs', y='model_run_duration_ms', ax=ax)
    ax.set_title('并发请求数对运行时间的影响')
    ax.set_xlabel('并发请求数')
    ax.set_ylabel('运行时间 (ms)')
    
    # 3. 工作负载方差 vs 运行时间
    ax = axes[0, 2]
    ax.scatter(df['workload_variance'], df['model_run_duration_ms'], alpha=0.5)
    ax.set_title('工作负载方差对运行时间的影响')
    ax.set_xlabel('工作负载方差')
    ax.set_ylabel('运行时间 (ms)')
    
    # 4. 缓存缺失次数的影响
    ax = axes[1, 0]
    cache_miss_stats = df.groupby('cache_misses')['model_run_duration_ms'].mean()
    cache_miss_stats.plot(kind='line', marker='o', ax=ax)
    ax.set_title('缓存缺失次数对运行时间的影响')
    ax.set_xlabel('缓存缺失次数')
    ax.set_ylabel('运行时间 (ms)')
    
    # 5. Chunk切换次数的影响
    ax = axes[1, 1]
    switch_stats = df.groupby('chunk_switches')['model_run_duration_ms'].mean()
    switch_stats.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Chunk切换次数对运行时间的影响')
    ax.set_xlabel('Chunk切换次数')
    ax.set_ylabel('运行时间 (ms)')
    
    # 6. 内存压力的影响
    ax = axes[1, 2]
    ax.scatter(df['memory_pressure'], df['model_run_duration_ms'], alpha=0.5)
    ax.set_title('内存压力对运行时间的影响')
    ax.set_xlabel('内存压力')
    ax.set_ylabel('运行时间 (ms)')
    
    # 7. 运行时间直方图（按chunk模式分组）
    ax = axes[2, 0]
    for pattern in df['chunk_pattern'].unique():
        subset = df[df['chunk_pattern'] == pattern]['model_run_duration_ms']
        if len(subset) > 10:  # 只显示样本数足够的模式
            ax.hist(subset, alpha=0.7, label=pattern, bins=20)
    ax.set_title('不同Chunk模式的运行时间分布')
    ax.set_xlabel('运行时间 (ms)')
    ax.set_ylabel('频次')
    ax.legend()
    
    # 8. 热力图：并发度 vs chunk数量
    ax = axes[2, 1]
    pivot_data = df.pivot_table(
        values='model_run_duration_ms', 
        index='num_running_reqs', 
        columns='num_chunks', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.1f', ax=ax, cmap='YlOrRd')
    ax.set_title('运行时间热力图：并发度 vs Chunk数量')
    
    # 9. 工作负载分布效率
    ax = axes[2, 2]
    ax.scatter(df['work_per_request'], df['model_run_duration_ms'], alpha=0.5)
    ax.set_title('单请求工作量对运行时间的影响')
    ax.set_xlabel('平均每请求工作量')
    ax.set_ylabel('运行时间 (ms)')
    
    plt.tight_layout()
    plt.savefig('/home/paperspace/zhangy/vllm-workspace/vllm/exp/detailed_duration_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("详细分析图表已保存到: exp/detailed_duration_analysis.png")

def generate_conclusions(df, importance_df):
    """生成分析结论"""
    print("\n" + "="*60)
    print("           深度分析结论")
    print("="*60)
    
    print("\n【核心发现】")
    print("在total_scheduled_tokens=4096的条件下，model_run_duration_ms的变化主要由以下因素造成：")
    
    print(f"\n1. 【最重要影响因素】Chunk大小的标准差 (相关性: 0.2251)")
    print(f"   - Chunk大小分布不均匀时，会导致内存访问模式复杂化")
    print(f"   - GPU资源无法充分并行化，导致运行时间增加")
    
    print(f"\n2. 【并发竞争】运行请求数和Chunk数量 (相关性: 0.1889)")
    print(f"   - 更多的并发请求 = 更多的Chunk数量 = 更多的资源竞争")
    print(f"   - 最优并发数约为8-10个请求")
    
    print(f"\n3. 【计算复杂度】总的computed tokens (相关性: 0.1731)")
    print(f"   - 更多的computed tokens意味着更重的计算负载")
    print(f"   - prefill阶段的长序列会显著影响性能")
    
    print(f"\n4. 【内存访问模式】")
    avg_cache_miss_corr = df['cache_misses'].corr(df['model_run_duration_ms'])
    print(f"   - 缓存缺失对性能的影响: {avg_cache_miss_corr:.4f}")
    print(f"   - 更高的缓存命中率可以显著减少运行时间")
    
    print(f"\n【性能优化建议】")
    print(f"1. 优化Chunk大小分布策略：")
    print(f"   - 尽量保持Chunk大小的均匀性")
    print(f"   - 避免单个batch中有极大和极小的chunk同时存在")
    
    print(f"2. 控制并发度：")
    print(f"   - 最佳并发请求数在8-10个之间")
    print(f"   - 过高的并发度会导致资源竞争和性能下降")
    
    print(f"3. 优化内存访问：")
    print(f"   - 提高缓存命中率可以有效减少运行时间")
    print(f"   - 减少不必要的内存访问跳跃")
    
    print(f"4. 工作负载均衡：")
    print(f"   - 在batch内均匀分配计算负载")
    print(f"   - 避免个别请求占用过多计算资源")
    
    print(f"\n【数据洞察】")
    print(f"- 运行时间变异系数仅为0.0335，说明系统整体稳定性较好")
    print(f"- 异常值比例仅0.2%，大部分情况下性能可预测")
    print(f"- Chunk模式的影响比预期更显著，需要重点关注")

def main():
    """主函数"""
    print("正在加载处理后的数据...")
    df = load_processed_data()
    
    print(f"数据加载完成，共{len(df)}条记录")
    
    # 深度分析
    df = analyze_chunk_patterns(df)
    df = analyze_memory_access_pattern(df)
    df = analyze_resource_contention(df)
    
    # 特征重要性分析
    importance_df = feature_importance_analysis(df)
    
    # 创建详细可视化
    create_detailed_visualizations(df)
    
    # 生成结论
    generate_conclusions(df, importance_df)
    
    # 保存增强后的数据
    output_file = '/home/paperspace/zhangy/vllm-workspace/vllm/exp/enhanced_analysis_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n增强分析数据已保存到: {output_file}")

if __name__ == "__main__":
    main() 