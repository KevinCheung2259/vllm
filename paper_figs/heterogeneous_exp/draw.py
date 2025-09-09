import os
import glob
import matplotlib.pyplot as plt
import argparse
import json
import re
import numpy as np
from csv_process import csv_process_file,csv_process_dir


def draw_image(dir,e2e_slo):

    # 算法名称及其对应配置
    algorithm_config = {
        'latency_based': {'name': 'latency based', 'color': 'blue', 'marker': 'o', 'linestyle': '--'}, 
        'least_loaded': {'name': 'least loaded', 'color': 'green', 'marker': 's', 'linestyle': '--'},
        'sla_elrar': {'name': 'sla elrar', 'color': 'red', 'marker': '^','linestyle': '-'}, 
        'weight_based': {'name': 'weight based', 'color': 'purple', 'marker': 'D', 'linestyle': '--'}
    }

    # 指标配置：定义需要绘制的性能指标及其可视化参数
    # 每个指标包含名称、Y轴标签、输出文件后缀和可选的Y轴范围
    metrics_config = [
        {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50'},
        {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90'},
        {'name': 'p95', 'label': 'P95 latency(s)', 'suffix': 'p95'},
        {'name': 'p99', 'label': 'P99 latency(s)', 'suffix': 'p99'},
        {'name': 'slo attainment', 'label': f'SLO attainment({e2e_slo}s) (%)', 'suffix': 'slo', 'y_lim': (0, 100)}
    ]

    # 初始化数据存储结构：按算法名称组织，存储不同指标的数值列表
    data = {}
    for alg_name in algorithm_config.keys():
        data[alg_name] = {
            'x_metrics': [],  # 根据数据类型动态使用QPS或采样率
            'p50': [], 'p90': [], 'p95': [], 'p99': [], 'slo attainment': []
        }

    dataset = os.path.basename(dir)

    # 采样率模式处理逻辑
    if dataset == 'Flowgpt-timestamp' or dataset == 'Flowgpt-qps':

        # 选择需要显示的采样率或者qps
        select_sampling_rates = [0.1, 0.12, 0.14, 0.16]
        select_qps = [5,7,9,11]

        # 遍历每种算法目录
        for alg_name, _ in algorithm_config.items():
            alg_dir = os.path.join(dir, alg_name)
            # 跳过不存在算法
            if not os.path.isdir(alg_dir):
                continue
            
            # 遍历算法目录下的子目录（每个子目录对应一个采样率）
            for dir_name in os.listdir(alg_dir):
                dir_path = os.path.join(alg_dir, dir_name)
                if not os.path.isdir(dir_path):

                    continue
                
                x_metrics = 0
                if dataset == 'Flowgpt-timestamp':
                    # 从文件夹名称提取采样率
                    sampling_rate_match = re.search(r'_(\d+\.\d+)$', dir_name)
                    if not sampling_rate_match:
                        continue

                    sampling_rate = float(sampling_rate_match.group(1))
                    if sampling_rate not in select_sampling_rates:
                        # 跳过不打印的采样率
                        continue
                    x_metrics = sampling_rate
                else:
                    # 从文件夹名称提取QPS值
                    qps_match = re.search(r'qps_(?:weight_based|sla_elrar|least_loaded|latency_based)_?(\d+)(?:_cv.*)?$', dir_name)
                    if not qps_match:
                        continue

                    qps = int(qps_match.group(1))
                    if qps not in select_qps:
                        continue
                    x_metrics = qps
                # 判断是否存在csv文件
                csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                if not csv_files:
                    continue

                try:
                    metrics = csv_process_dir(dir_path, e2e_slo)
                    data[alg_name]['x_metrics'].append(x_metrics)
                    data[alg_name]['p50'].append(metrics['p50'])
                    data[alg_name]['p90'].append(metrics['p90'])
                    data[alg_name]['p95'].append(metrics['p95'])
                    data[alg_name]['p99'].append(metrics['p99'])
                    data[alg_name]['slo attainment'].append(metrics['slo attainment'])
                except Exception as e:
                    print(f'处理 {alg_name} 采样率={sampling_rate} 时出错: {str(e)}')
                    continue

        # 按 x_metrics 排序
        for alg_name in data:
            if data[alg_name]['x_metrics']:
                sorted_pairs = sorted(zip(
                    data[alg_name]['x_metrics'],
                    data[alg_name]['p50'],
                    data[alg_name]['p90'],
                    data[alg_name]['p95'],
                    data[alg_name]['p99'],
                    data[alg_name]['slo attainment']
                ), key=lambda x: x[0])
                data[alg_name]['x_metrics'] = [p[0] for p in sorted_pairs]
                data[alg_name]['p50'] = [p[1] for p in sorted_pairs]
                data[alg_name]['p90'] = [p[2] for p in sorted_pairs]
                data[alg_name]['p95'] = [p[3] for p in sorted_pairs]
                data[alg_name]['p99'] = [p[4] for p in sorted_pairs]
                data[alg_name]['slo attainment'] = [p[5] for p in sorted_pairs]

    else:
        for alg_name, _ in algorithm_config.items():
            dir_path = os.path.join(dir, alg_name)
            
            # 判断是否存在算法目录
            if not os.path.exists(dir_path):
                continue

            csv_files = glob.glob(os.path.join(dir_path, '*.csv'))
            if not csv_files:
                continue
            
            select_qps = [5,7,9,11]

            
            algorithm_metrics = []

            for file in csv_files:
                filename = os.path.basename(file)
                qps_match = re.search(r'_(\d+)\.csv$', filename)
                if not qps_match:
                    continue
                qps = int(qps_match.group(1))
                if qps not in select_qps:
                    continue

                try:
                    metrics = csv_process_file(file, e2e_slo)
                    algorithm_metrics.append((qps, metrics['p50'], metrics['p90'], metrics['p95'], metrics['p99'], metrics['slo attainment']))
                except Exception as e:
                    continue

            if algorithm_metrics:
                algorithm_metrics.sort(key=lambda x: x[0])
                data[alg_name]['x_metrics'] = [item[0] for item in algorithm_metrics]
                data[alg_name]['p50'] = [item[1] for item in algorithm_metrics]
                data[alg_name]['p90'] = [item[2] for item in algorithm_metrics]
                data[alg_name]['p95'] = [item[3] for item in algorithm_metrics]
                data[alg_name]['p99'] = [item[4] for item in algorithm_metrics]
                data[alg_name]['slo attainment'] = [item[5] for item in algorithm_metrics]

    # 生成图表
    for metric in metrics_config:
        plot_metric(data, metric['name'], metric['label'], metric['suffix'],
                    algorithm_config,dataset, y_lim=metric.get('y_lim'))

def plot_metric(data, metric_name, y_label, output_suffix, algorithm_config,dataset, y_lim=None):
    # if dataset == 'Flowgpt-timestamp' or dataset == 'Flowgpt-qps':
    #     if metric_name == "slo attainment":
    #         plt.figure(figsize=(7.5, 6))
    #     else:
    #         plt.figure(figsize=(6, 6))
    # else:
    #     if metric_name == "slo attainment":
    #         plt.figure(figsize=(7.5, 5))
    #     else:
    #         plt.figure(figsize=(6, 5))
    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.95)
    for alg_name, config in algorithm_config.items():
        name = config['name']
        alg_data = data.get(alg_name, {})
        if alg_data.get('x_metrics') and alg_data.get(metric_name):
            plt.plot(alg_data['x_metrics'], alg_data[metric_name], marker=config['marker'],
                     linestyle=config['linestyle'], color=config['color'], linewidth=4, markersize=10, label=name)

    # 收集所有唯一的数据点用于x轴刻度
    all_data_points = []
    for algorithm_name, alg_data in data.items():
        if alg_data['x_metrics']:
            all_data_points.extend(alg_data['x_metrics'])
    unique_data_points = sorted(list(set(all_data_points)))

    # 动态设置x轴标签
    if dataset == 'Flowgpt-timestamp':
        x_label = 'Sample Rate'
        plt.xlabel(x_label, fontsize=20)
    elif dataset == 'Flowgpt-qps':
        x_label = 'Request Rate'
        plt.xlabel(x_label, fontsize=20)
    # elif dataset in ['Flowgpt-qps', 'Coder', 'ShareGPT']:
    #     x_label = 'QPS'
    # plt.xlabel(x_label, fontsize=20)
    if metric_name == "slo attainment":
        plt.ylabel(y_label, fontsize=20)

    plt.grid(True, linestyle='--', alpha=0.7,linewidth=1.5)
    # plt.legend(fontsize=20, loc='best') # 图例
    plt.xticks(unique_data_points, fontsize=25)

    plt.yticks(fontsize=25)
    if y_lim:
        plt.ylim(y_lim)
    # plt.tight_layout()

    output_dir =  f"./picture/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path =f"{output_dir}/{dataset}_{output_suffix}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'图片已保存至: {os.path.abspath(output_path)}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='整合QPS和采样率模式的图表生成工具')
    parser.add_argument('--dir', type=str, default='/home/paperspace/cys/projects/exp_3/result/Flowgpt-timestamp',
                        help='处理的数据集路径')
    parser.add_argument('--e2e-slo', type=float, default=9.5)
    args = parser.parse_args()
    draw_image(args.dir,args.e2e_slo)