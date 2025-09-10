#!/usr/bin/env python3
"""
vLLM Scheduler 时间趋势图生成工具

这个脚本专门用于生成 vLLM scheduler 的时间趋势比较图（Time Trend Comparison），
显示调度时间和模型运行时间随批次变化的趋势。
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体支持（如果需要）
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_and_process_data(log_file_or_dir='scheduler_profiling.jsonl'):
    """加载并处理profiling数据"""
    log_path = Path(log_file_or_dir)
    
    # 如果是目录，查找其中的jsonl文件
    if log_path.is_dir():
        jsonl_files = list(log_path.glob('*.jsonl'))
        if not jsonl_files:
            print(f"❌ 在目录 {log_file_or_dir} 中没有找到jsonl文件")
            return None
        # 使用全部jsonl文件
        log_files = jsonl_files
        print(f"📁 找到目录: {log_file_or_dir}")
        print(f"📄 使用文件: {len(jsonl_files)} 个")
    else:
        log_file = log_path
        if not log_file.exists():
            print(f"❌ 日志文件 {log_file} 不存在")
            return None
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
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            # 去掉每个文件的前10行和后10行，避免文件开头和结尾可能的非JSON内容
            lines = f.readlines()
            for line in lines[10:-10]:
                try:
                    entry = json.loads(line.strip())
                    # 对batch id进行重新处理，确保连续性
                    if 'batch_id' in entry:
                        entry['batch_id'] += batch_id_offset
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
        batch_id_offset += len(lines[10:-10])

    # 数据清洗
    # 将调度时间在300ms以上的数据删除，过滤异常值
    data = [item for item in data if item['schedule_duration_ms'] < 300]
    # 将运行时间在200ms以上的数据删除，过滤异常值
    data = [item for item in data if item['model_run_duration_ms'] < 200]
    
    if not data:
        print("❌ 没有找到有效的profiling数据")
        return None
    
    df = pd.DataFrame(data)
    print(f"✅ 成功读取 {len(data)} 条profiling数据")
    
    # 基本统计信息
    print("\n📊 Scheduler 时间统计")
    print("=" * 50)
    print(f"⏱️  平均调度时间: {df['schedule_duration_ms'].mean():.2f}ms")
    print(f"⏱️  最大调度时间: {df['schedule_duration_ms'].max():.2f}ms")
    print(f"⏱️  最小调度时间: {df['schedule_duration_ms'].min():.2f}ms")
    
    if 'model_run_duration_ms' in df.columns:
        print(f"⚡ 平均Model Run时间: {df['model_run_duration_ms'].mean():.2f}ms")
        print(f"⚡ 最大Model Run时间: {df['model_run_duration_ms'].max():.2f}ms")
        print(f"⚡ 最小Model Run时间: {df['model_run_duration_ms'].min():.2f}ms")
    
    print(f"🔢 平均总Token数: {df['total_scheduled_tokens'].mean():.2f}")
    
    return df

def create_time_trend_plot(df, output_file='./time_trend_comparison.png'):

    """创建时间趋势比较图，支持添加模拟数据"""
    # 检查是否有model run时间数据
    has_model_run_data = 'model_run_duration_ms' in df.columns
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 绘制时间趋势
    if has_model_run_data:
        ax.plot(df['batch_id'], df['schedule_duration_ms'], label='Engine schedule Time', alpha=0.7)
        ax.plot(df['batch_id'], df['model_run_duration_ms'], label='Model Run Time', alpha=0.7)
        # 计算总时间并绘制（如果需要）
        # total_time = df['schedule_duration_ms'] + df['model_run_duration_ms']
        # ax.plot(df['batch_id'], total_time, label='Total Time', alpha=0.7)
        ## 标题
        # ax.set_title('Time Trend Comparison')
    else:
        ax.plot(df['batch_id'], df['schedule_duration_ms'], label='Engine schedule Time')
        ax.set_title('Schedule Time Trend')
    
    # 添加图例（如果有多个数据系列）
    if has_model_run_data or (sim_mean is not None and sim_variance is not None):
        ax.legend(fontsize = 12,markerscale=1.1)
    
    # 设置坐标轴标签和网格
    ax.set_xlabel('Batch ID',fontsize=16,labelpad=10)
    ax.set_ylabel('Time (ms)',fontsize=16,labelpad=10)
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', pad=8, labelsize=12)  
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.subplots_adjust(left=0.15, right=0.8, bottom=0.15, top=0.8)
    # 美化图表
    plt.tight_layout()
    pdf_path = "./sechdule_overhead.pdf"
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n📊 时间趋势图已保存为: {output_file}")
    
    # 显示图表（如果在交互环境中运行）
    # plt.show()

def main():
    
    print("🚀 vLLM Scheduler 时间趋势图生成工具")
    print("=" * 40)

    log_dir = "./../../exp"
    parser = argparse.ArgumentParser(description='生成 Schedule Overhead 图表')
    parser.add_argument('log_path', type=str, nargs='*',default=f"{log_dir}/profiling_result_a100",
                      help='profiling数据文件或目录路径 (可指定多个，默认: profiling_result)')
    parser.add_argument('--save-path', type=str, default="./sechdule_overhead.png")
    
    args = parser.parse_args()
    
    log_path = args.log_path
    output_file = args.save_path
    
    # 加载和处理数据
    df = load_and_process_data(log_path)
    if df is None:
        return
    
    # 创建并保存时间趋势图，传入模拟数据参数和密度因子
    create_time_trend_plot(df, output_file)
    
    print("\n✅ 处理完成!")

if __name__ == '__main__':
    main()