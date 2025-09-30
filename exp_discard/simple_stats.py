#!/usr/bin/env python3
"""
简单统计脚本 - 计算平均model_run_time和model_scheduler_time
"""

import json
import sys
from pathlib import Path

def load_data(file_path):
    """加载数据"""
    data = []
    
    if Path(file_path).is_dir():
        # 目录处理
        jsonl_files = list(Path(file_path).glob('*.jsonl'))
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        data.append(entry)
                    except json.JSONDecodeError:
                        continue
    else:
        # 单文件处理
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return data

def calculate_stats(data):
    """计算统计信息"""
    if not data:
        return None
    
    model_run_times = []
    scheduler_times = []
    
    for entry in data:
        if 'model_run_duration_ms' in entry:
            model_run_times.append(entry['model_run_duration_ms'])
        if 'schedule_duration_ms' in entry:
            scheduler_times.append(entry['schedule_duration_ms'])
    
    stats = {}
    if model_run_times:
        stats['model_run'] = {
            'count': len(model_run_times),
            'mean': sum(model_run_times) / len(model_run_times),
            'min': min(model_run_times),
            'max': max(model_run_times)
        }
    
    if scheduler_times:
        stats['scheduler'] = {
            'count': len(scheduler_times),
            'mean': sum(scheduler_times) / len(scheduler_times),
            'min': min(scheduler_times),
            'max': max(scheduler_times)
        }
    
    return stats

def main():
    if len(sys.argv) != 3:
        print("使用方法: python simple_stats.py file1 file2")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    print("📊 加载数据...")
    data1 = load_data(file1_path)
    data2 = load_data(file2_path)
    
    print(f"文件1数据点数: {len(data1)}")
    print(f"文件2数据点数: {len(data2)}")
    
    print("\n📈 统计结果:")
    print("=" * 60)
    
    # 文件1统计
    stats1 = calculate_stats(data1)
    if stats1:
        print(f"\n📁 {file1_path}:")
        if 'model_run' in stats1:
            m = stats1['model_run']
            print(f"  Model Run Time: 平均={m['mean']:.2f}ms, 最小={m['min']:.2f}ms, 最大={m['max']:.2f}ms, 数量={m['count']}")
        if 'scheduler' in stats1:
            s = stats1['scheduler']
            print(f"  Scheduler Time: 平均={s['mean']:.2f}ms, 最小={s['min']:.2f}ms, 最大={s['max']:.2f}ms, 数量={s['count']}")
    
    # 文件2统计
    stats2 = calculate_stats(data2)
    if stats2:
        print(f"\n📁 {file2_path}:")
        if 'model_run' in stats2:
            m = stats2['model_run']
            print(f"  Model Run Time: 平均={m['mean']:.2f}ms, 最小={m['min']:.2f}ms, 最大={m['max']:.2f}ms, 数量={m['count']}")
        if 'scheduler' in stats2:
            s = stats2['scheduler']
            print(f"  Scheduler Time: 平均={s['mean']:.2f}ms, 最小={s['min']:.2f}ms, 最大={s['max']:.2f}ms, 数量={s['count']}")
    
    # 对比
    if stats1 and stats2:
        print(f"\n📊 对比:")
        if 'model_run' in stats1 and 'model_run' in stats2:
            diff = stats2['model_run']['mean'] - stats1['model_run']['mean']
            print(f"  Model Run Time差异: {diff:+.2f}ms")
        if 'scheduler' in stats1 and 'scheduler' in stats2:
            diff = stats2['scheduler']['mean'] - stats1['scheduler']['mean']
            print(f"  Scheduler Time差异: {diff:+.2f}ms")

if __name__ == '__main__':
    main()
