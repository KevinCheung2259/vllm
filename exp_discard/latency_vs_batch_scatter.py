#!/usr/bin/env python3
"""
OSDI投稿用散点图绘制脚本
绘制在固定Total Token数量下，Latency随Batch Size变化的散点图

使用方法:
python osdi_latency_vs_batch_scatter.py <数据文件或目录路径>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

# 设置matplotlib参数以获得OSDI论文级别的高质量输出
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.unicode_minus': False,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

def load_profiling_data(log_file_or_dir):
    """加载profiling数据 - 从paper_heatmap.py复用"""
    log_path = Path(log_file_or_dir)
    
    # 确定要处理的文件列表
    if log_path.is_dir():
        jsonl_files = list(log_path.glob('*.jsonl'))
        if not jsonl_files:
            raise FileNotFoundError(f"在目录 {log_file_or_dir} 中没有找到jsonl文件")
        log_files = jsonl_files
        print(f"📁 找到目录: {log_file_or_dir}, 文件数: {len(jsonl_files)}")
    else:
        if not log_path.exists():
            raise FileNotFoundError(f"日志文件 {log_file_or_dir} 不存在")
        log_files = [log_path]
        print(f"📄 使用单个文件: {log_path}")
    
    # 读取和合并数据
    data = []
    batch_id_offset = 0
    
    # 按文件名排序
    if len(log_files) > 1:
        try:
            log_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        except Exception:
            log_files.sort(key=lambda x: x.name)
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过前后10行以避免不稳定数据
            for line in lines[10:-10]:
                try:
                    entry = json.loads(line.strip())
                    if 'batch_id' in entry:
                        entry['batch_id'] += batch_id_offset
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
        batch_id_offset += len(lines[10:-10])
    
    # 数据清洗：移除异常值
    print(f"原始数据点数: {len(data)}")
    data = [item for item in data if item.get('schedule_duration_ms', 0) < 300]
    data = [item for item in data if item.get('model_run_duration_ms', 0) < 200]
    print(f"清洗后数据点数: {len(data)}")
    
    if not data:
        raise ValueError("没有找到有效的profiling数据")
    
    return pd.DataFrame(data)

def extract_features_for_scatter(df):
    """为散点图提取特征：batch_size, total_tokens, latency"""
    def get_batch_size(sizes):
        if isinstance(sizes, list):
            return len(sizes)
        return np.nan
    
    def get_total_tokens(sizes):
        if isinstance(sizes, list):
            return sum(sizes)
        if isinstance(sizes, (int, float)):
            return float(sizes)
        return np.nan
    
    df['batch_size'] = df['chunk_sizes'].apply(get_batch_size)
    df['total_tokens'] = df['chunk_sizes'].apply(get_total_tokens)
    df['latency_ms'] = df['model_run_duration_ms']
    
    # 过滤有效数据
    valid_df = df[
        (df['batch_size'] >= 1) & (df['batch_size'] <= 120) &
        (df['total_tokens'] >= 50) & (df['total_tokens'] <= 8192) &
        (df['latency_ms'].notna()) & (df['latency_ms'] > 0)
    ].copy()
    
    print(f"提取特征后有效数据点数: {len(valid_df)}")
    print(f"Total tokens范围: {valid_df['total_tokens'].min():.0f} - {valid_df['total_tokens'].max():.0f}")
    print(f"Batch size范围: {valid_df['batch_size'].min():.0f} - {valid_df['batch_size'].max():.0f}")
    print(f"Latency范围: {valid_df['latency_ms'].min():.2f} - {valid_df['latency_ms'].max():.2f} ms")
    
    return valid_df

def filter_data_by_token_ranges(valid_df, target_tokens=[128, 256, 512, 1024, 2048, 4096], tolerance=0.15):
    """
    根据目标token数量过滤数据
    tolerance: 允许的相对误差范围，例如0.15表示±15%
    """
    filtered_data = {}
    
    for target in target_tokens:
        # 计算容忍范围
        lower_bound = target * (1 - tolerance)
        upper_bound = target * (1 + tolerance)
        
        # 过滤数据
        mask = (valid_df['total_tokens'] >= lower_bound) & (valid_df['total_tokens'] <= upper_bound)
        filtered_subset = valid_df[mask].copy()
        
        if len(filtered_subset) > 0:
            filtered_data[target] = filtered_subset
            print(f"Token={target}: 找到 {len(filtered_subset)} 个数据点 (范围: {lower_bound:.0f}-{upper_bound:.0f})")
        else:
            print(f"Token={target}: 未找到数据点 (范围: {lower_bound:.0f}-{upper_bound:.0f})")
    
    return filtered_data

def compute_statistics(filtered_data):
    """计算每个token级别和batch size的统计信息"""
    stats_data = {}
    
    for token_count, data in filtered_data.items():
        # 按batch_size分组并计算统计信息
        grouped = data.groupby('batch_size')['latency_ms'].agg([
            'mean', 'std', 'count', 'median'
        ]).reset_index()
        
        # 只保留有足够数据点的组（至少3个数据点）
        grouped = grouped[grouped['count'] >= 3]
        
        if len(grouped) > 0:
            stats_data[token_count] = grouped
            print(f"Token={token_count}: {len(grouped)} 个有效batch size组")
    
    return stats_data

def plot_latency_vs_batch_scatter(stats_data, filtered_data, output_path='osdi_latency_vs_batch_scatter.pdf'):
    """绘制OSDI投稿用的高质量散点图"""
    
    # 定义颜色方案 - 使用专业的颜色搭配
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    token_counts = sorted(stats_data.keys())
    
    for i, token_count in enumerate(token_counts):
        if token_count not in stats_data:
            continue
            
        stats = stats_data[token_count]
        raw_data = filtered_data[token_count]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # 绘制原始数据点（半透明）
        ax.scatter(raw_data['batch_size'], raw_data['latency_ms'], 
                  alpha=0.3, s=20, c=color, marker=marker)
        
        # 绘制统计均值点（突出显示）
        ax.scatter(stats['batch_size'], stats['mean'], 
                  s=80, c=color, marker=marker, 
                  label=f'Total Tokens = {token_count}', 
                  edgecolors='black', linewidth=1)
        
        # 添加误差线
        ax.errorbar(stats['batch_size'], stats['mean'], yerr=stats['std'], 
                   fmt='none', ecolor=color, alpha=0.7, capsize=3)
    
    # 设置图表属性
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Latency vs Batch Size for Different Total Token Counts', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 设置坐标轴范围
    ax.set_xlim(0, max([data['batch_size'].max() for data in filtered_data.values()]) + 5)
    ax.set_ylim(0, max([data['latency_ms'].max() for data in filtered_data.values()]) * 1.1)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存为多种格式
    base_name = output_path.replace('.pdf', '')
    plt.savefig(f'{base_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_name}.eps', format='eps', dpi=300, bbox_inches='tight')
    
    print(f"✅ OSDI散点图已保存:")
    print(f"   PDF: {base_name}.pdf")
    print(f"   PNG: {base_name}.png") 
    print(f"   EPS: {base_name}.eps")
    
    plt.show()

def generate_summary_statistics(stats_data, filtered_data):
    """生成汇总统计信息"""
    print("\n📊 数据汇总统计:")
    print("=" * 60)
    
    for token_count in sorted(stats_data.keys()):
        stats = stats_data[token_count]
        raw_data = filtered_data[token_count]
        
        print(f"\nTotal Tokens = {token_count}:")
        print(f"  数据点总数: {len(raw_data)}")
        print(f"  Batch size范围: {raw_data['batch_size'].min():.0f} - {raw_data['batch_size'].max():.0f}")
        print(f"  平均Latency: {raw_data['latency_ms'].mean():.2f} ± {raw_data['latency_ms'].std():.2f} ms")
        print(f"  Latency范围: {raw_data['latency_ms'].min():.2f} - {raw_data['latency_ms'].max():.2f} ms")
        
        # 分析batch size对latency的影响
        if len(stats) >= 3:
            batch_sizes = stats['batch_size'].values
            latencies = stats['mean'].values
            
            # 计算相关系数
            correlation = np.corrcoef(batch_sizes, latencies)[0, 1]
            print(f"  Batch size与Latency相关系数: {correlation:.3f}")
            
            # 计算增长率（从最小到最大batch size）
            if len(latencies) >= 2:
                growth_rate = (latencies[-1] - latencies[0]) / latencies[0] * 100
                print(f"  Latency增长率: {growth_rate:.1f}%")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python osdi_latency_vs_batch_scatter.py <数据文件或目录路径>")
        print("示例: python osdi_latency_vs_batch_scatter.py profiling_result")
        return
    
    data_path = sys.argv[1]
    
    try:
        print("🚀 开始生成OSDI投稿用散点图")
        print("=" * 50)
        
        # 1. 加载数据
        df = load_profiling_data(data_path)
        
        # 2. 提取特征
        valid_df = extract_features_for_scatter(df)
        
        if len(valid_df) < 50:
            print("⚠️ 有效数据点较少，可能影响图表质量")
        
        # 3. 按token数量过滤数据
        target_tokens = [128, 256, 512, 1024, 2048, 4096]
        filtered_data = filter_data_by_token_ranges(valid_df, target_tokens)
        
        if not filtered_data:
            print("❌ 没有找到符合条件的数据，请检查数据文件")
            return
        
        # 4. 计算统计信息
        stats_data = compute_statistics(filtered_data)
        
        # 5. 绘制散点图
        plot_latency_vs_batch_scatter(stats_data, filtered_data)
        
        # 6. 生成汇总统计
        generate_summary_statistics(stats_data, filtered_data)
        
        print("\n🎉 OSDI投稿图表生成完成！")
        print("=" * 50)
        print("📊 主要输出文件:")
        print("   - osdi_latency_vs_batch_scatter.pdf (投稿用主图)")
        print("   - osdi_latency_vs_batch_scatter.png (预览用)")
        print("   - osdi_latency_vs_batch_scatter.eps (高质量矢量图)")
        print("\n📝 OSDI投稿建议:")
        print("   - 图表显示了在固定token数量下batch size对latency的影响")
        print("   - 误差线表示标准差，展示了数据的变异性")
        print("   - 半透明点显示原始数据分布，实心点显示统计均值")
        print("   - 建议在论文中讨论batch size的饱和效应")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 