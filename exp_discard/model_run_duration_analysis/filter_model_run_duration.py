#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

def filter_model_run_duration(input_file, output_file, threshold=100):
    """
    筛选CSV文件中model_run_duration_ms大于指定阈值的记录
    并保留指定的列：batch_id, timestamp, total_scheduled_tokens, chunk_sizes, all_computed_tokens, model_run_duration_ms
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        threshold (float): model_run_duration_ms的阈值，默认为115
    """
    
    print(f"正在读取文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列数: {len(df.columns)}")
    
    # 筛选model_run_duration_ms > threshold的记录
    filtered_df = df[df['model_run_duration_ms'] > threshold]
    
    print(f"筛选后数据行数: {len(filtered_df)}")
    
    # 选择需要的列
    columns_to_keep = [
        'batch_id', 
        'timestamp', 
        'total_scheduled_tokens', 
        'chunk_sizes', 
        'all_computed_tokens', 
        'model_run_duration_ms'
    ]
    
    # 检查所需列是否存在
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        print(f"警告：以下列在原数据中不存在: {missing_columns}")
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    # 选择指定的列
    result_df = filtered_df[columns_to_keep].copy()
    
    # 保存到新文件
    result_df.to_csv(output_file, index=False)
    
    print(f"筛选完成！")
    print(f"保留的列: {columns_to_keep}")
    print(f"输出文件: {output_file}")
    print(f"输出数据行数: {len(result_df)}")
    
    # 显示筛选结果的一些统计信息
    if len(result_df) > 0:
        print(f"\nmodel_run_duration_ms统计信息:")
        print(f"最小值: {result_df['model_run_duration_ms'].min():.2f}")
        print(f"最大值: {result_df['model_run_duration_ms'].max():.2f}")
        print(f"平均值: {result_df['model_run_duration_ms'].mean():.2f}")
        print(f"中位数: {result_df['model_run_duration_ms'].median():.2f}")

if __name__ == "__main__":
    # 文件路径
    input_file = "/home/paperspace/zhangy/vllm-workspace/vllm/exp/model_run_duration_analysis/filtered_data_4096.csv"
    output_file = "/home/paperspace/zhangy/vllm-workspace/vllm/exp/model_run_duration_analysis/filtered_data_4096_high_duration.csv"
    
    # 执行筛选
    filter_model_run_duration(input_file, output_file, threshold=110) 