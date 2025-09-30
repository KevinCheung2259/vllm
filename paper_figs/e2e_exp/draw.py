import os
import glob
import matplotlib.pyplot as plt
import argparse
import json
import re
import numpy as np

from matplotlib.ticker import MaxNLocator

from csv_process import csv_process_dir

def get_data(dir,e2e_slo,algorithm_config = None,
            select_sampling_rates:list = None,select_qps:list = None,data_fiter:list = None):

    # 初始化数据存储结构：按算法名称组织，存储不同指标的数值列表
    data = {}
    dataset = os.path.basename(dir)

    for alg_name in algorithm_config.keys():
        data[alg_name] = {
            'x_metrics': [],  # 根据数据类型动态使用QPS或采样率
            'p50': [], 'p90': [], 'p95': [], 'p99': [], 'slo attainment': [],'ttft':[],'tpot':[]
        }

    # 遍历每种算法目录
    for alg_name, _ in algorithm_config.items():
        alg_dir = os.path.join(dir, alg_name)
        
        # 判断该数据集下是否存在此算法目录，不存在则跳过
        if not os.path.isdir(alg_dir):
            continue
        
        # 遍历该算法目录下的每个数据子目录（每个子目录对应一个采样率或qps的数据）
        for data_dir in os.listdir(alg_dir):
            data_path = os.path.join(alg_dir, data_dir)

            # 不是目录则跳过
            if not os.path.isdir(data_path):
                continue
            
            x_metrics = 0

            # 采样率模式
            if dataset == 'flowgpt_timestamp':
                # 从文件夹名称提取采样率
                sampling_rate_match = re.search(r'_(\d+\.\d+)$', data_dir)
                if not sampling_rate_match:
                    # 提取不到采样率则跳过
                    continue
                
                sampling_rate = float(sampling_rate_match.group(1))
                if sampling_rate not in select_sampling_rates:
                    # 跳过不打印的采样率
                    continue
                x_metrics = int(50*sampling_rate)
            # qps 模式
            else:
                # 从文件夹名称提取QPS值
                qps_match = re.search(r'qps_(?:native_roundrobin|native_session|sla_elrar|v0_roundrobin|sla_roundrobin|v0_session|sla_session|native_elrar)_?(\d+)(?:_cv.*)?$', data_dir)
                if not qps_match:
                    # 提取不到qps则跳过
                    print("提取不到qps目录")
                    continue

                qps = int(qps_match.group(1))
                if qps not in select_qps:
                    # 跳过不打印的qps
                    continue
                x_metrics = qps

            # 判断是否存在csv文件
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if not csv_files:
                continue
            
            # 处理数据集文件
            metrics = csv_process_dir(data_path,e2e_slo,dataset,data_fiter)

            data[alg_name]['x_metrics'].append(x_metrics) # x 坐标
            data[alg_name]['p50'].append(metrics['p50'])
            data[alg_name]['p90'].append(metrics['p90'])
            data[alg_name]['p95'].append(metrics['p95'])
            data[alg_name]['p99'].append(metrics['p99'])
            data[alg_name]['slo attainment'].append(metrics['slo attainment'])
            data[alg_name]['ttft'].append(metrics['ttft'])
            data[alg_name]['tpot'].append(metrics['tpot'])

    # 获取完该数据集的所有数据,按 x_metrics 排序
    for alg_name in data:
        if data[alg_name]['x_metrics']:
            sorted_pairs = sorted(zip(
                data[alg_name]['x_metrics'],
                data[alg_name]['p50'],
                data[alg_name]['p90'],
                data[alg_name]['p95'],
                data[alg_name]['p99'],
                data[alg_name]['slo attainment'],
                data[alg_name]['ttft'],
                data[alg_name]['tpot']
            ), key=lambda x: x[0])
            data[alg_name]['x_metrics'] = [p[0] for p in sorted_pairs]
            data[alg_name]['p50'] = [p[1] for p in sorted_pairs]
            data[alg_name]['p90'] = [p[2] for p in sorted_pairs]
            data[alg_name]['p95'] = [p[3] for p in sorted_pairs]
            data[alg_name]['p99'] = [p[4] for p in sorted_pairs]
            data[alg_name]['slo attainment'] = [p[5] for p in sorted_pairs]
            data[alg_name]['ttft'] = [p[6] for p in sorted_pairs]
            data[alg_name]['tpot'] = [p[7] for p in sorted_pairs]

    return data

def plot_metric(data,metrics_config, algorithm_config,dataset):
    # 大小
    plt.figure(figsize=(4.5, 3))

    # 绘图区域大小设置
    plt.subplots_adjust(left=0.2, right=0.96, bottom=0.15, top=0.95)
    

    for alg_name, config in algorithm_config.items():
        name = config['name']
        alg_data = data.get(alg_name, {})
        if alg_data.get('x_metrics') and alg_data.get(metrics_config['name']):
            if config['linestyle'] == '--':
                dashes =[5,3]
            else:
                dashes = [1,0]
            plt.plot(alg_data['x_metrics'], alg_data[metrics_config['name']], marker=config['marker'],
                     linestyle=config['linestyle'], color=config['color'], linewidth=3, markersize=15, label=name,dashes = dashes)

    # 收集所有唯一的数据点用于x轴刻度
    all_data_points = []
    for algorithm_name, alg_data in data.items():
        if alg_data['x_metrics']:
            all_data_points.extend(alg_data['x_metrics'])
    unique_data_points = sorted(list(set(all_data_points)))

    # # 动态设置x轴标签,labelpad:与x轴距离
    # if dataset == 'reasoning':
    #     plt.xlabel('Request Rate', fontsize=25,labelpad=5)

    # if metrics_config['name'] == "slo attainment":
    #     plt.ylabel(metrics_config['label'], fontsize=20)

    # plt.ylabel(metrics_config['label'], fontsize=20)


    plt.grid(True, linestyle='--', alpha=0.7,linewidth=1.5)
    # 图例设置
    # plt.legend(fontsize=14, loc='best')

    plt.xticks(unique_data_points, fontsize=25)
    
    # Y轴刻度设置
    plt.yticks(fontsize=25)
    if metrics_config.get('custom_yticks'):
        plt.yticks(metrics_config['custom_yticks'], fontsize=25)
    elif metrics_config.get('y_lim'):
        plt.ylim(metrics_config['y_lim'])
        # 设置Y轴刻度强制至少占两格
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    else:
        # 设置Y轴刻度强制至少占两格
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

     # 添加橙色横线（如果指定了横线纵坐标值）
    if metrics_config.get('horizontal_line_y'):
        plt.axhline(y=metrics_config['horizontal_line_y'], color='orange', linestyle='--', linewidth=3,dashes = [4,7])

    # 这个参数是自动调整布局的，慎用
    # plt.tight_layout()


    # 获取当前文件的完整路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    output_dir =  f"{current_dir}/picture/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path =f"{output_dir}/{dataset}_{metrics_config['suffix']}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f'图片已保存至: {os.path.abspath(output_path)}')


def draw(dir,e2e_slo=None,metrics_configs = None,data_fiter = None):

    dataset = os.path.basename(dir)

    e2e_slos = {
        "arxiv":15,
        "sharegpt":6,
        "reasoning":6,
        "flowgpt_qps":10,
        "flowgpt_timestamp":20
    }

    if not e2e_slo:
        e2e_slo = e2e_slos[dataset]

    # 需要修改的

    # 算法的画图配置
    algorithm_config = {
        'native_session': {'name': 'Sarathi Session', 'color': '#9933FF', 'marker': 'o', 'linestyle': '--'}, 
        'v0_roundrobin': {'name': 'vLLM RR', 'color': '#3399FF', 'marker': 's', 'linestyle': '--'},
        'sla_elrar': {'name': 'SynergySched', 'color': 'red', 'marker': '^','linestyle': '-'}, 
        'native_roundrobin': {'name': 'Sarathi RR', 'color': '#994C00', 'marker': 'D', 'linestyle': '--'},
        # 'sla_roundrobin': {'name': 'LENS RR', 'color': 'orange', 'marker': 'x', 'linestyle': '--'},
        'v0_session': {'name': 'vLLM Session', 'color': '#00CC00', 'marker': '*', 'linestyle': '--'},
        # 'sla_session': {'name': 'LENS Session', 'color': 'magenta', 'marker': 'p', 'linestyle': '--'},
        # 'native_elrar': {'name': 'Sarathi PRISM', 'color': '#CCCC00', 'marker': 'h', 'linestyle': '--'}
    }

    # p50,p90等指标的配置
    if not metrics_configs:
        metrics_configs = [
            {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50','title':'E2E p50'},
            {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','title':'E2E p90'},
            {'name': 'p95', 'label': 'P95 latency(s)', 'suffix': 'p95','title':'E2E p95'},
            {'name': 'p99', 'label': 'P99 latency(s)', 'suffix': 'p99','title':'E2E p99'},
            {'name': 'slo attainment', 'label': f'SLO attainment({e2e_slo}s) (%)', 'suffix': 'slo', 'y_lim': (70, 100),'title':'SLO Attainment'},
            {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','title':'TTFT'},
            {'name': 'tpot', 'label': 'TPOT (s)', 'suffix': 'tpot','title':'TPOT'}
        ]

    select_sampling_rates = [0.28,0.30,0.32,0.34,0.36]

    select_qps = {
        "arxiv":[12,13,14,15,16],
        "sharegpt":[40,44,48,52],
        "reasoning":[50,52,54,56,58],
        "flowgpt_qps":[14,15,16,17,18],
        "flowgpt_timestamp":None
    }

    # 读取数据集下所有数据
    data = get_data(dir,e2e_slo,algorithm_config,select_sampling_rates,select_qps[dataset],data_fiter)

    # 画图
    for metrics_config in metrics_configs:
        plot_metric(data, metrics_config,algorithm_config,dataset)

if __name__ == '__main__':

    # 读取命令行参数
    parser = argparse.ArgumentParser(description='针对端到端实验的画图工具')
    parser.add_argument('--dir', type=str, default=f'{os.getcwd()}/data/sharegpt',help='数据集路径')
    parser.add_argument('--e2e-slo', type=float)
    args = parser.parse_args()
    
    draw(args.dir,args.e2e_slo)
