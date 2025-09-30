#!/usr/bin/env python3
"""
EOS Token概率预测分析脚本

这个脚本用于分析从vLLM调度器记录的EOS token概率数据，
探索是否能够根据EOS概率变化预测请求何时结束。
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd


def load_eos_data(log_file: str = 'eos_probabilities.jsonl') -> List[Dict[str, Any]]:
    """加载EOS概率数据"""
    if not os.path.exists(log_file):
        print(f"未找到日志文件: {log_file}")
        return []
    
    entries = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        print(f"成功加载 {len(entries)} 条EOS概率记录")
        return entries
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return []


def group_by_request(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按请求ID分组数据"""
    request_data = {}
    for entry in entries:
        req_id = entry['request_id']
        if req_id not in request_data:
            request_data[req_id] = []
        request_data[req_id].append(entry)
    
    # 按步骤排序
    for req_id in request_data:
        request_data[req_id].sort(key=lambda x: x['step'])
    
    return request_data


def analyze_eos_trends(request_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """分析EOS概率变化趋势"""
    analysis = {
        'total_requests': len(request_data),
        'finished_requests': 0,
        'avg_sequence_length': 0,
        'eos_prob_patterns': []
    }
    
    sequence_lengths = []
    
    for req_id, data in request_data.items():
        if not data:
            continue
        
        # 基本统计
        sequence_length = len(data)
        sequence_lengths.append(sequence_length)
        
        is_finished = data[-1].get('is_finished', False)
        if is_finished:
            analysis['finished_requests'] += 1
        
        # EOS概率序列
        eos_probs = []
        for entry in data:
            prob = entry.get('eos_prob')
            if prob is not None:
                eos_probs.append(prob)
            else:
                eos_probs.append(0.0)  # 如果没有记录到EOS概率，设为0
        
        if eos_probs:
            pattern = {
                'request_id': req_id,
                'length': sequence_length,
                'eos_probs': eos_probs,
                'final_eos_prob': eos_probs[-1],
                'max_eos_prob': max(eos_probs),
                'avg_eos_prob': np.mean(eos_probs),
                'eos_prob_trend': np.polyfit(range(len(eos_probs)), eos_probs, 1)[0] if len(eos_probs) > 1 else 0,
                'is_finished': is_finished,
                'finish_reason': data[-1].get('finish_reason')
            }
            analysis['eos_prob_patterns'].append(pattern)
    
    if sequence_lengths:
        analysis['avg_sequence_length'] = np.mean(sequence_lengths)
    
    return analysis


def visualize_eos_trends(analysis: Dict[str, Any], save_plots: bool = True):
    """可视化EOS概率趋势"""
    patterns = analysis['eos_prob_patterns']
    if not patterns:
        print("没有可用的EOS概率数据进行可视化")
        return
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EOS Token概率分析', fontsize=16)
    
    # 1. 显示几个典型请求的EOS概率变化
    ax1 = axes[0, 0]
    for i, pattern in enumerate(patterns[:5]):  # 显示前5个请求
        steps = range(len(pattern['eos_probs']))
        ax1.plot(steps, pattern['eos_probs'], 
                label=f"请求 {i+1}", marker='o', markersize=4)
    ax1.set_xlabel('生成步骤')
    ax1.set_ylabel('EOS概率')
    ax1.set_title('典型请求的EOS概率变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最终EOS概率分布
    ax2 = axes[0, 1]
    final_probs = [p['final_eos_prob'] for p in patterns]
    ax2.hist(final_probs, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('最终EOS概率')
    ax2.set_ylabel('请求数量')
    ax2.set_title('最终EOS概率分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. EOS概率趋势 vs 序列长度
    ax3 = axes[1, 0]
    trends = [p['eos_prob_trend'] for p in patterns]
    lengths = [p['length'] for p in patterns]
    colors = ['red' if p['is_finished'] else 'blue' for p in patterns]
    scatter = ax3.scatter(lengths, trends, c=colors, alpha=0.6)
    ax3.set_xlabel('序列长度')
    ax3.set_ylabel('EOS概率趋势（斜率）')
    ax3.set_title('EOS概率趋势 vs 序列长度')
    ax3.grid(True, alpha=0.3)
    
    # 添加图例
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=8, label='已结束')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                           markersize=8, label='未结束')
    ax3.legend(handles=[red_patch, blue_patch])
    
    # 4. 平均EOS概率 vs 最大EOS概率
    ax4 = axes[1, 1]
    avg_probs = [p['avg_eos_prob'] for p in patterns]
    max_probs = [p['max_eos_prob'] for p in patterns]
    ax4.scatter(avg_probs, max_probs, alpha=0.6, c=colors)
    ax4.set_xlabel('平均EOS概率')
    ax4.set_ylabel('最大EOS概率')
    ax4.set_title('平均EOS概率 vs 最大EOS概率')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('eos_probability_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为: eos_probability_analysis.png")
    
    plt.show()


def predict_termination(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """尝试预测请求终止的简单模型"""
    # 收集特征和标签
    features = []
    labels = []
    
    for pattern in patterns:
        if len(pattern['eos_probs']) < 3:  # 需要至少3个数据点
            continue
        
        # 特征工程：使用前几步的统计信息
        eos_probs = pattern['eos_probs']
        n_steps = min(5, len(eos_probs))  # 使用前5步或全部数据
        
        feature_vector = [
            np.mean(eos_probs[:n_steps]),     # 前n步平均EOS概率
            np.max(eos_probs[:n_steps]),      # 前n步最大EOS概率
            np.std(eos_probs[:n_steps]),      # 前n步EOS概率标准差
            eos_probs[-1] if eos_probs else 0, # 当前EOS概率
            len(eos_probs),                   # 当前序列长度
        ]
        
        features.append(feature_vector)
        labels.append(1 if pattern['is_finished'] else 0)
    
    if len(features) < 5:
        return {'error': '数据不足，无法建立预测模型'}
    
    # 简单的统计分析
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    finished_mask = labels_array == 1
    unfinished_mask = labels_array == 0
    
    analysis = {
        'total_samples': len(features),
        'finished_samples': np.sum(finished_mask),
        'unfinished_samples': np.sum(unfinished_mask),
    }
    
    # 计算特征的区分能力
    if np.sum(finished_mask) > 0 and np.sum(unfinished_mask) > 0:
        feature_names = ['avg_eos_prob', 'max_eos_prob', 'std_eos_prob', 'current_eos_prob', 'sequence_length']
        
        for i, name in enumerate(feature_names):
            finished_mean = np.mean(features_array[finished_mask, i])
            unfinished_mean = np.mean(features_array[unfinished_mask, i])
            
            analysis[f'{name}_finished_mean'] = finished_mean
            analysis[f'{name}_unfinished_mean'] = unfinished_mean
            analysis[f'{name}_difference'] = finished_mean - unfinished_mean
    
    return analysis


def print_detailed_analysis(analysis: Dict[str, Any]):
    """打印详细分析结果"""
    print("=== 详细分析结果 ===")
    print(f"总请求数: {analysis['total_requests']}")
    print(f"已完成请求数: {analysis['finished_requests']}")
    print(f"完成率: {analysis['finished_requests']/analysis['total_requests']*100:.1f}%")
    print(f"平均序列长度: {analysis['avg_sequence_length']:.1f}")
    
    patterns = analysis['eos_prob_patterns']
    if patterns:
        final_probs = [p['final_eos_prob'] for p in patterns]
        max_probs = [p['max_eos_prob'] for p in patterns]
        avg_probs = [p['avg_eos_prob'] for p in patterns]
        trends = [p['eos_prob_trend'] for p in patterns]
        
        print(f"\n=== EOS概率统计 ===")
        print(f"最终EOS概率范围: {min(final_probs):.4f} - {max(final_probs):.4f}")
        print(f"平均最终EOS概率: {np.mean(final_probs):.4f}")
        print(f"最大EOS概率范围: {min(max_probs):.4f} - {max(max_probs):.4f}")
        print(f"平均EOS概率范围: {min(avg_probs):.4f} - {max(avg_probs):.4f}")
        print(f"EOS概率趋势范围: {min(trends):.6f} - {max(trends):.6f}")
        
        # 按完成状态分组分析
        finished_patterns = [p for p in patterns if p['is_finished']]
        unfinished_patterns = [p for p in patterns if not p['is_finished']]
        
        if finished_patterns and unfinished_patterns:
            print(f"\n=== 已完成 vs 未完成请求对比 ===")
            finished_final = np.mean([p['final_eos_prob'] for p in finished_patterns])
            unfinished_final = np.mean([p['final_eos_prob'] for p in unfinished_patterns])
            print(f"已完成请求平均最终EOS概率: {finished_final:.4f}")
            print(f"未完成请求平均最终EOS概率: {unfinished_final:.4f}")
            
            finished_trend = np.mean([p['eos_prob_trend'] for p in finished_patterns])
            unfinished_trend = np.mean([p['eos_prob_trend'] for p in unfinished_patterns])
            print(f"已完成请求平均EOS概率趋势: {finished_trend:.6f}")
            print(f"未完成请求平均EOS概率趋势: {unfinished_trend:.6f}")


def main():
    """主函数"""
    print("=== EOS Token概率预测分析 ===")
    
    # 加载数据
    log_file = 'eos_probabilities.jsonl'
    entries = load_eos_data(log_file)
    
    if not entries:
        print("没有数据可分析")
        return
    
    # 按请求分组
    request_data = group_by_request(entries)
    print(f"共分析 {len(request_data)} 个请求")
    
    # 分析EOS概率趋势
    analysis = analyze_eos_trends(request_data)
    
    # 打印详细分析
    print_detailed_analysis(analysis)
    
    # 可视化
    print("\n正在生成可视化图表...")
    visualize_eos_trends(analysis)
    
    # 预测分析
    print("\n=== 终止预测分析 ===")
    prediction_analysis = predict_termination(analysis['eos_prob_patterns'])
    
    if 'error' in prediction_analysis:
        print(prediction_analysis['error'])
    else:
        print(f"分析样本数: {prediction_analysis['total_samples']}")
        print(f"已完成样本: {prediction_analysis['finished_samples']}")
        print(f"未完成样本: {prediction_analysis['unfinished_samples']}")
        
        # 显示特征区分能力
        feature_names = ['avg_eos_prob', 'max_eos_prob', 'std_eos_prob', 'current_eos_prob', 'sequence_length']
        print("\n特征区分能力分析:")
        for name in feature_names:
            finished_key = f'{name}_finished_mean'
            unfinished_key = f'{name}_unfinished_mean'
            diff_key = f'{name}_difference'
            
            if finished_key in prediction_analysis:
                print(f"  {name}:")
                print(f"    已完成平均值: {prediction_analysis[finished_key]:.4f}")
                print(f"    未完成平均值: {prediction_analysis[unfinished_key]:.4f}")
                print(f"    差异: {prediction_analysis[diff_key]:.4f}")
    
    print("\n=== 分析完成 ===")
    print("基于这些数据，您可以:")
    print("1. 观察EOS概率的变化模式")
    print("2. 分析哪些特征最能预测请求结束")
    print("3. 建立更复杂的机器学习模型进行预测")


if __name__ == "__main__":
    main() 