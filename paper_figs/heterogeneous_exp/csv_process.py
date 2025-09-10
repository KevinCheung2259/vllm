import pandas as pd
import argparse
import glob
import os
import numpy as np
from datetime import datetime

"""
    包含两种处理csv数据的格式
    目录:针对使用 online_replay.py 生成的数据
    文件：针对使用 multi-round-qa.py 生成的数据
"""

# 输入csv文件的上一层路径
def csv_process_dir(path,e2e_slo):

    csv_files = glob.glob(os.path.join(path, '*.csv'))

    if not csv_files:
        return
    
    # 保存所有数据
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        
        df['latency'] = df['total_time'] - df['send_time']
        
        df_sorted = df.sort_values('send_time')
        
        # 过滤数据
        min_time = df_sorted['send_time'].min()
        max_time = df_sorted['send_time'].max()

        filtered_df = df_sorted[
            (df_sorted['send_time'] > min_time + 30) &  
            (df_sorted['send_time'] < max_time - 30) 
        ]
        
        all_data.append(filtered_df)
    
    # 合并
    data = pd.concat(all_data, ignore_index=True)

    return {
        'p50': data['latency'].quantile(0.5),
        'p90': data['latency'].quantile(0.9),
        'p95': data['latency'].quantile(0.95),
        'p99': data['latency'].quantile(0.99),
        'total_requests': len(data),
        'slo_attainment_requests':(data['latency'] <= e2e_slo).sum(),
        'slo attainment': ((data['latency'] <= e2e_slo).sum() / len(data)) * 100,
        'actual_qps':len(data)/data['send_time'].max() - data['send_time'].min()
    }

# 输入具体的某个csv文件路径
def csv_process_file(file_path, e2e_slo):
    df = pd.read_csv(file_path)

    # 按 launch_time 排序
    df_sort = df.sort_values('launch_time')
    
    min_time = df_sort['launch_time'].min()
    max_time = df_sort['launch_time'].max()

    data = df_sort[(df_sort['launch_time'] > min_time + 30) & 
                 (df_sort['launch_time'] < max_time -30)].copy()
    
    # 计算延迟指标
    data.loc[:,'latency'] = data.loc[:,'finish_time'] - data.loc[:,'launch_time']
    
    return {
        'p50': data['latency'].quantile(0.5), 
        'p90': data['latency'].quantile(0.9),
        'p95': data['latency'].quantile(0.95),
        'p99': data['latency'].quantile(0.99),
        'total_requests': len(data),
        'slo_attainment_requests': (data['latency'] <= e2e_slo).sum(),
        'slo attainment': ((data['latency'] <= e2e_slo).sum() / len(data)) * 100,
        'actual_qps':len(data)/(data['launch_time'].max() - data['launch_time'].min())
    }

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',required=True,type=str)
    parser.add_argument('--e2e-slo', type=float, default=5)
    args = parser.parse_args()

    if args.path.endswith('.csv'):
        metrics = csv_process_file(args.path,args.e2e_slo)
    else:
        metrics = csv_process_dir(args.path,args.e2e_slo)

    print(f"处理 {args.path} 目录下数据")
    print(f"一共处理 {metrics['total_requests']} 个 请求")
    print(f"Actual QPS: {metrics['actual_qps']:.2f} r/s")
    print(f"P50 Latency: {metrics['p50']:.4f} s")
    print(f"P90 Latency: {metrics['p90']:.4f} s")
    print(f"P95 Latency: {metrics['p95']:.4f} s")
    print(f"P99 Latency: {metrics['p99']:.4f} s")
    print(f"Slo attainment(Latency<={args.e2e_slo}s): {metrics['slo attainment']:.2f}% ({metrics['slo_attainment_requests']}/{metrics['total_requests']})\n")