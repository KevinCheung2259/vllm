#!/usr/bin/env python3
"""
vLLM Scheduler Profiling 示例脚本

这个脚本展示了如何启用和使用vLLM scheduler的profiling功能来分析调度性能。
"""

import os
import subprocess
import sys
import json
from pathlib import Path
import numpy as np
import math

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请安装依赖: pip install pandas matplotlib")
    sys.exit(1)

def enable_scheduler_profiling():
    """启用scheduler profiling"""
    # 设置环境变量来启用profiling
    os.environ['VLLM_ENABLE_SCHEDULER_PROFILING'] = 'true'
    os.environ['VLLM_SCHEDULER_PROFILING_LOG'] = 'scheduler_profiling.jsonl'
    os.environ['VLLM_SCHEDULER_PROFILING_CONSOLE'] = 'true'
    
    print("✅ Scheduler Profiling 已启用")
    print(f"📝 日志文件: {os.environ['VLLM_SCHEDULER_PROFILING_LOG']}")
    print(f"📟 控制台输出: 已启用")

def analyze_profiling_data(log_file_or_dir='profiling_result'):
    """分析profiling数据"""
    log_path = Path(log_file_or_dir)
    
    # 如果是目录，查找其中的jsonl文件
    if log_path.is_dir():
        jsonl_files = list(log_path.glob('*.jsonl'))
        if not jsonl_files:
            print(f"❌ 在目录 {log_file_or_dir} 中没有找到jsonl文件")
            return
        # 使用全部jsonl文件
        log_files = jsonl_files
        print(f"📁 找到目录: {log_file_or_dir}")
        print(f"📄 使用文件: {len(jsonl_files)} 个")
    else:
        log_file = log_path
        if not log_file.exists():
            print(f"❌ 日志文件 {log_file} 不存在")
            return
        # 单文件分析
        log_files = [log_file]
        print(f"📄 使用单个文件: {log_file}")
    
    # 读取数据
    data = []
    batch_id_offset = 0
    # 根据文件名中的数字排序；若只有一个文件或解析失败则按名称排序/保持不变
    if len(log_files) > 1:
        try:
            log_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        except Exception:
            log_files.sort(key=lambda x: x.name)
    # print(log_files)
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            # 去掉每个文件的前10行和后10行
            lines = f.readlines()
            for line in lines[10:-10]:
                try:
                    entry = json.loads(line.strip())
                    # 对batch id进行重新处理
                    if 'batch_id' in entry:
                        entry['batch_id'] += batch_id_offset
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
        batch_id_offset += len(lines[10:-10])

    # 数据清洗
    # 将调度时间在300ms以上的数据删除
    data = [item for item in data if item['schedule_duration_ms'] < 300]
    # # 将运行时间在200ms以上的数据删除
    data = [item for item in data if item['model_run_duration_ms'] < 200]
    
    if not data:
        print("❌ 没有找到有效的profiling数据")
        return
    
    df = pd.DataFrame(data)
    print(f"✅ 成功读取 {len(data)} 条profiling数据")
    
    # 基于 chunk_sizes 中值为 1 的个数计算 Decode 请求数
    if 'chunk_sizes' in df.columns:
        def _count_decode_reqs(sizes):
            if isinstance(sizes, list):
                return sum(1 for s in sizes if s == 1)
            try:
                return 1 if sizes == 1 else 0
            except Exception:
                return 0
        df['num_decode_reqs'] = df['chunk_sizes'].apply(_count_decode_reqs)

        def _count_prefill_reqs(sizes):
            if isinstance(sizes, list):
                return sum(1 for s in sizes if s > 1)
            try:
                return 1 if sizes > 1 else 0
            except Exception:
                return 0
        df['num_prefill_reqs'] = df['chunk_sizes'].apply(_count_prefill_reqs)

    # 基本统计信息
    print("\n📊 Scheduler Profiling 分析报告")
    print("=" * 50)
    print(f"📈 总批次数: {len(df)}")
    print(f"⏱️  平均调度时间: {df['schedule_duration_ms'].mean():.2f}ms")
    print(f"⏱️  最大调度时间: {df['schedule_duration_ms'].max():.2f}ms")
    print(f"⏱️  最小调度时间: {df['schedule_duration_ms'].min():.2f}ms")
    
    if 'model_run_duration_ms' in df.columns:
        print(f"⚡ 平均Model Run时间: {df['model_run_duration_ms'].mean():.2f}ms")
        print(f"⚡ 最大Model Run时间: {df['model_run_duration_ms'].max():.2f}ms")
        print(f"⚡ 最小Model Run时间: {df['model_run_duration_ms'].min():.2f}ms")
    
    # 请求数统计（支持新旧数据格式）
    if 'num_prefill_reqs' in df.columns:
        print(f"\n🔢 平均Prefill请求数: {df['num_prefill_reqs'].mean():.2f}")
    if 'num_decode_reqs' in df.columns:
        print(f"🔢 平均Decode请求数: {df['num_decode_reqs'].mean():.2f}")
    
    print(f"🔢 平均总Token数: {df['total_scheduled_tokens'].mean():.2f}")
    
    # Chunk size分析（如果存在）
    if 'chunk_sizes' in df.columns:
        # 计算平均chunk size
        chunk_sizes = []
        for sizes in df['chunk_sizes']:
            if isinstance(sizes, list) and sizes:
                chunk_sizes.extend(sizes)
        if chunk_sizes:
            print(f"\n📦 平均Chunk Size: {sum(chunk_sizes)/len(chunk_sizes):.2f}")
    
    # 新增：KV cache hit分析
    if 'all_cached_tokens' in df.columns:
        # 计算KV cache统计
        all_cached_tokens = []
        for cached_tokens in df['all_cached_tokens']:
            if isinstance(cached_tokens, list):
                # 过滤掉-1值（未设置的cached tokens）
                valid_cached = [t for t in cached_tokens if t >= 0]
                if valid_cached:
                    all_cached_tokens.extend(valid_cached)
        
        if all_cached_tokens:
            print(f"\n🗂️  KV Cache统计:")
            print(f"   平均每请求缓存命中: {sum(all_cached_tokens)/len(all_cached_tokens):.2f}")
            print(f"   最大缓存命中数: {max(all_cached_tokens)}")
            print(f"   缓存命中率(按请求): {len([t for t in all_cached_tokens if t > 0])/len(all_cached_tokens)*100:.1f}%")
    
    # 新增：computed tokens分析
    if 'all_computed_tokens' in df.columns:
        all_computed_tokens = []
        for computed_tokens in df['all_computed_tokens']:
            if isinstance(computed_tokens, list):
                all_computed_tokens.extend(computed_tokens)
        
        if all_computed_tokens:
            print(f"\n🔄 Computed Tokens统计:")
            print(f"   平均已计算Token数: {sum(all_computed_tokens)/len(all_computed_tokens):.2f}")
            print(f"   最大已计算Token数: {max(all_computed_tokens)}")
            print(f"   最小已计算Token数: {min(all_computed_tokens)}")
    
    # 创建可视化图表
    create_profiling_plots(df)

def create_profiling_plots(df):
    """创建profiling数据的可视化图表"""
    # 移除中文字体设置，使用默认英文字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 检查是否有model run时间数据和cached tokens数据
    has_model_run_data = 'model_run_duration_ms' in df.columns
    has_cached_data = 'all_cached_tokens' in df.columns
    
    # 动态调整子图布局
    if has_model_run_data and has_cached_data:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    elif has_model_run_data or has_cached_data:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    fig.suptitle('vLLM Scheduler Profiling Analysis', fontsize=16)
    
    # 时间分布
    if has_model_run_data:
        axes[0, 0].plot(df['batch_id'], df['schedule_duration_ms'], label='Schedule Time', alpha=0.7)
        axes[0, 0].plot(df['batch_id'], df['model_run_duration_ms'], label='Model Run Time', alpha=0.7)
        # 计算总时间（如果存在的话）
        # if 'total_step_duration_ms' in df.columns:
        #     axes[0, 0].plot(df['batch_id'], df['total_step_duration_ms'], label='Total Time', alpha=0.7)
        # else:
        #     total_time = df['schedule_duration_ms'] + df['model_run_duration_ms']
        #     axes[0, 0].plot(df['batch_id'], total_time, label='Total Time', alpha=0.7)
        axes[0, 0].set_title('Time Trend Comparison')
        axes[0, 0].legend()
    else:
        axes[0, 0].plot(df['batch_id'], df['schedule_duration_ms'])
        axes[0, 0].set_title('Schedule Time Trend')
    axes[0, 0].set_xlabel('Batch ID')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].grid(True)
    
    # Prefill vs Decode请求数（如果数据存在）
    ax_idx = (0, 1)
    axes[ax_idx].plot(df['batch_id'], df['num_prefill_reqs'], label='Prefill Requests')
    axes[ax_idx].plot(df['batch_id'], df['num_decode_reqs'], label='Decode Requests')
    axes[ax_idx].set_xlabel('Batch ID')
    axes[ax_idx].set_ylabel('Number of Requests')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True)

    # chunk_sizes vs 运行时间分布
    ax_idx = (0, 2)
    # 处理chunk_sizes数据 - 计算每个批次的总chunk size
    prefill_chunk_totals = df['chunk_sizes'].apply(lambda x: sum(x) if isinstance(x, list) else x)
    
    # 根据chunk_sizes的长度设置颜色，长度越大颜色越深
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    chunk_lengths = []
    for chunk_sizes in df['chunk_sizes']:
        if isinstance(chunk_sizes, list):
            chunk_lengths.append(len(chunk_sizes))
        else:
            chunk_lengths.append(1)  # 如果不是列表，默认长度为1
    
    # 创建颜色映射，从浅到深
    if chunk_lengths:
        min_len = min(chunk_lengths)
        max_len = max(chunk_lengths)
        if max_len > min_len:
            # 使用Blues颜色映射，数值越大颜色越深
            norm = mcolors.Normalize(vmin=min_len, vmax=max_len)
            cmap = cm.Blues
            colors = [cmap(norm(length)) for length in chunk_lengths]
        else:
            colors = ['blue'] * len(chunk_lengths)  # 所有长度相同时使用统一颜色
    else:
        colors = ['blue']  # 默认颜色
    axes[ax_idx].scatter(prefill_chunk_totals, df['model_run_duration_ms'], c=colors, alpha=0.6, s=20)

    # # 处理chunk_sizes数据 - 计算chunk_sizes中大于1的个数，大于1的为橙色
    
    # # 计算每个批次的总scheduled tokens
    # total_scheduled_tokens = df['total_scheduled_tokens']
    
    # # 根据chunk_sizes中大于1的元素个数来着色
    # colors = []
    # for chunk_sizes in df['chunk_sizes']:
    #     if isinstance(chunk_sizes, list):
    #         # 统计chunk_sizes中大于1的元素个数
    #         large_chunks_count = sum(1 for size in chunk_sizes if size > 1)
    #         if large_chunks_count > 1:
    #             colors.append('orange')  # 有大于1的chunk使用橙色
    #         else:
    #             colors.append('blue')    # 全为1的chunk使用蓝色
    #     else:
    #         # 如果不是列表，根据数值判断
    #         colors.append('orange' if chunk_sizes > 1 else 'blue')
    
    # # 绘制散点图
    # axes[ax_idx].scatter(total_scheduled_tokens, df['model_run_duration_ms'], 
    #                     c=colors, alpha=0.6, s=20)
    
    axes[ax_idx].set_title('Num Scheduled Tokens vs Model Run Time')
    axes[ax_idx].set_xlabel('Num Scheduled Tokens')
    axes[ax_idx].set_ylabel('Model Run Time (ms)')
    axes[ax_idx].grid(True)
    
    # 拟合：线性与二次多项式，并给出函数表达式与R^2
    try:
        x_series = pd.to_numeric(prefill_chunk_totals, errors='coerce')
        y_series = pd.to_numeric(df['model_run_duration_ms'], errors='coerce')
        valid_mask = x_series.notna() & y_series.notna()
        x = x_series[valid_mask].to_numpy()
        y = y_series[valid_mask].to_numpy()
        if x.size >= 3:
            # 线性拟合 y = m*x + c
            coeffs_lin = np.polyfit(x, y, 1)
            m, c = coeffs_lin[0], coeffs_lin[1]
            p_lin = np.poly1d(coeffs_lin)
            y_pred_lin = p_lin(x)
            ss_res_lin = np.sum((y - y_pred_lin) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0.0
            
            # 二次拟合 y = a*x^2 + b*x + c2
            coeffs_quad = np.polyfit(x, y, 2)
            a2, b2, c2 = coeffs_quad[0], coeffs_quad[1], coeffs_quad[2]
            p_quad = np.poly1d(coeffs_quad)
            y_pred_quad = p_quad(x)
            ss_res_quad = np.sum((y - y_pred_quad) ** 2)
            r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0.0
            
            # 选择更优模型（按R^2）并在图上叠加
            xs = np.linspace(x.min(), x.max(), 200)
            if r2_quad > r2_lin + 0.2:
                axes[ax_idx].plot(xs, p_quad(xs), color='red', linewidth=2, label=f'Quad Fit (R^2={r2_quad:.3f})')
                chosen = '二次'
            else:
                axes[ax_idx].plot(xs, p_lin(xs), color='red', linewidth=2, label=f'Linear Fit (R^2={r2_lin:.3f})')
                chosen = '线性'
            axes[ax_idx].legend()
            
            # 控制台输出拟合表达式
            print("\n📐 Chunk Size 与 Model Run Time 拟合表达式")
            print(f"线性: y = {c:.6f} + {m:.6f} * x, R^2 = {r2_lin:.6f}")
            print(f"二次: y = {c2:.6f} + {b2:.6f} * x + {a2:.6f} * x^2, R^2 = {r2_quad:.6f}")
            print(f"→ 选择: {chosen} 模型")
    except Exception as e:
        print(f"⚠️ 拟合失败: {e}")

    # num_decode_reqs vs 运行时间分布
    ax_idx = (1, 0)
    axes[ax_idx].scatter(df['num_decode_reqs'], df['model_run_duration_ms'], alpha=0.6, s=20)
    axes[ax_idx].set_title('Num Decode Requests vs Model Run Time')
    axes[ax_idx].set_xlabel('Num Decode Requests')
    axes[ax_idx].set_ylabel('Model Run Time (ms)')
    axes[ax_idx].grid(True)
    
    # num_prefill_reqs vs 运行时间分布
    ax_idx = (1, 1)
    axes[ax_idx].scatter(df['num_prefill_reqs'], df['model_run_duration_ms'], alpha=0.6, s=20)
    axes[ax_idx].set_title('Num Prefill Requests vs Model Run Time')
    axes[ax_idx].set_xlabel('Num Prefill Requests')
    axes[ax_idx].set_ylabel('Model Run Time (ms)')
    axes[ax_idx].grid(True)

    # 控制prefill_chunk_totals全为1，查看num_decode_reqs vs 运行时间分布
    ax_idx = (1, 2)
    num_decode_reqs = []
    model_run_duration_ms = []
    for _, row in df.iterrows():
        sizes = row.get('chunk_sizes', [])
        if isinstance(sizes, list) and sum(sizes) == len(sizes):
            num_decode_reqs.append(row['num_decode_reqs'])
            model_run_duration_ms.append(row['model_run_duration_ms'])
    axes[ax_idx].scatter(num_decode_reqs, model_run_duration_ms, alpha=0.6, s=20)
    axes[ax_idx].set_title('Decode Only Requests vs Model Run Time')
    axes[ax_idx].set_xlabel('Num Decode Requests')
    axes[ax_idx].set_ylabel('Model Run Time (ms)')
    axes[ax_idx].grid(True)

    # batch_size, chunk_size, model_run_duration_ms热力图
    ax_idx = (2, 0)
    try:
        if 'chunk_sizes' in df.columns and 'model_run_duration_ms' in df.columns:
            # 从chunk_sizes推导batch_size与平均chunk_size
            def _batch_size_from_chunks(sizes):
                if isinstance(sizes, list):
                    return len(sizes)
                return np.nan

            def _avg_chunk_size(sizes):
                if isinstance(sizes, list) and len(sizes) > 0:
                    return sum(sizes)
                if isinstance(sizes, (int, float)):
                    return float(sizes)
                return np.nan

            df['_batch_size_est'] = df['chunk_sizes'].apply(_batch_size_from_chunks)
            df['_chunk_size_est'] = df['chunk_sizes'].apply(_avg_chunk_size)

            # 过滤有效数据范围
            valid_df = df[
                (df['_batch_size_est'] >= 1) & (df['_batch_size_est'] <= 120) &
                (df['_chunk_size_est'] >= 1) & (df['_chunk_size_est'] <= 4096) &
                (df['model_run_duration_ms'].notna())
            ].copy()

            if not valid_df.empty:
                # 将数值分档，使用更合理的档位数量
                batch_bins = 12  # batch_size 1-120 分为12档
                chunk_bins = 16  # chunk_size 1-4096 分为16档
                
                # 创建分档边界
                batch_edges = np.linspace(1, 120, batch_bins + 1)
                chunk_edges = np.linspace(1, 4096, chunk_bins + 1)
                
                # 将数据分配到对应的档位
                valid_df['_batch_bin'] = pd.cut(valid_df['_batch_size_est'], bins=batch_edges, include_lowest=True, labels=False)
                valid_df['_chunk_bin'] = pd.cut(valid_df['_chunk_size_est'], bins=chunk_edges, include_lowest=True, labels=False)

                # 创建透视表
                pivot = valid_df.groupby(['_chunk_bin', '_batch_bin'])['model_run_duration_ms'].mean().unstack(fill_value=np.nan)

                # 确保所有档位都存在（填充缺失的档位）
                full_batch_range = range(batch_bins)
                full_chunk_range = range(chunk_bins)
                pivot = pivot.reindex(index=full_chunk_range, columns=full_batch_range)

                # 仅在3x3布局下绘制到(2,0)，否则跳过避免影响主图保存
                if hasattr(axes, 'shape') and axes.shape[0] >= 3 and axes.shape[1] >= 1:
                    ax_heat = axes[2, 0]
                    
                    # 使用masked array处理NaN值，这样没有数据的地方会显示为白色
                    masked_data = np.ma.masked_invalid(pivot.values)
                    
                    im = ax_heat.imshow(
                        masked_data, 
                        origin='lower', 
                        cmap='viridis',  # 使用viridis颜色映射，对缺失值更友好
                        aspect='auto',
                        interpolation='nearest'
                    )
                    
                    ax_heat.set_title('Heatmap: Batch Size vs Avg Chunk Size')
                    ax_heat.set_xlabel('Batch Size')
                    ax_heat.set_ylabel('Avg Chunk Size')
                    
                    # 设置坐标轴标签，显示实际的数值范围
                    batch_labels = [f'{int(batch_edges[i])}-{int(batch_edges[i+1])}' for i in range(0, len(batch_edges)-1, 2)]
                    chunk_labels = [f'{int(chunk_edges[i])}-{int(chunk_edges[i+1])}' for i in range(0, len(chunk_edges)-1, 3)]
                    
                    ax_heat.set_xticks(range(0, batch_bins, 2))
                    ax_heat.set_xticklabels(batch_labels, rotation=45, ha='right')
                    ax_heat.set_yticks(range(0, chunk_bins, 3))
                    ax_heat.set_yticklabels(chunk_labels)
                    
                    # 添加颜色条
                    cbar = plt.colorbar(im, ax=ax_heat)
                    cbar.set_label('Model Run Time (ms)')
                    
                    print(f"📊 热力图生成成功，数据点数: {len(valid_df)}")
                else:
                    print('ℹ️ 子图布局不足以放置热力图，已跳过绘制（需要3x3布局）。')
            else:
                print('⚠️ 在指定范围内未找到有效数据，无法生成热力图')
    except Exception as e:
        print(f"⚠️ 热力图绘制失败: {e}")
        import traceback
        traceback.print_exc()
  
    # 如果有KV cache数据，添加额外的图表
    if has_cached_data and 'all_cached_tokens' in df.columns:
        # 计算每批次的cache hit统计
        cache_hit_totals = []
        cache_hit_rates = []
        
        for idx, row in df.iterrows():
            cached_tokens = row.get('all_cached_tokens', [])
            if isinstance(cached_tokens, list):
                valid_cached = [t for t in cached_tokens if t >= 0]
                cache_hit_totals.append(sum(valid_cached))
                if valid_cached:
                    cache_hit_rates.append(len([t for t in valid_cached if t > 0]) / len(valid_cached) * 100)
                else:
                    cache_hit_rates.append(0)
        
        if cache_hit_totals:
            # Cache Hit总数趋势
            if has_model_run_data and has_cached_data and axes.shape[0] >= 3:
                # 如果有3x3布局，使用最后一行最后一列
                ax_idx = (2, 2)
                axes[ax_idx].plot(df['batch_id'], cache_hit_totals, color='green')
                axes[ax_idx].set_title('KV Cache Hit Tokens Trend')
                axes[ax_idx].set_xlabel('Batch ID')
                axes[ax_idx].set_ylabel('Cache Hit Tokens')
                axes[ax_idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('scheduler_profiling_analysis_a100.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 图表已保存为: scheduler_profiling_analysis_a100.png")

def main():
    """主函数"""
    print("🚀 vLLM Scheduler Profiling 工具")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("""
使用方法:
  python scheduler_profiling_example.py enable                    # 启用profiling
  python scheduler_profiling_example.py analyze                   # 分析profiling_result目录中的数据
  python scheduler_profiling_example.py analyze <file_or_dir>     # 分析指定文件或目录中的数据
        """)
        return
    
    command = sys.argv[1]
    
    if command == 'enable':
        enable_scheduler_profiling()
        print("""
🔧 现在你可以运行vLLM服务器:

export VLLM_ENABLE_SCHEDULER_PROFILING=true
export VLLM_SCHEDULER_PROFILING_LOG=scheduler_profiling.jsonl
export VLLM_SCHEDULER_PROFILING_CONSOLE=true

python -m vllm.entrypoints.openai.api_server \\
    --model your_model_name \\
    --host 0.0.0.0 \\
    --port 8000

然后发送请求进行测试，profiling数据将被记录到 scheduler_profiling.jsonl 文件中。

🔍 分析数据时，请使用:
python scheduler_profiling_example.py analyze profiling_result
        """)
    
    elif command == 'analyze':
        log_path = sys.argv[2] if len(sys.argv) > 2 else 'profiling_result'
        analyze_profiling_data(log_path)
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == '__main__':
    main() 