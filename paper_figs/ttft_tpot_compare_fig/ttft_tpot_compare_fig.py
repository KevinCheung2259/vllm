import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import argparse

# ==== 统一为与 latency_comparison 相近的论文风格（sans-serif） ====

def apply_topconf_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
        'axes.linewidth': 1.2,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'figure.dpi': 160,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        # 轻网格、路径优化
        'grid.alpha': 0.25,
        'path.simplify': True,
        'path.simplify_threshold': 0.5,
        'agg.path.chunksize': 10000,
    })

# 颜色参考论文图（蓝/橙）
COLOR_TTFT = 'orange'
COLOR_TPOT = 'blue'


def data_process(log_dir):

    chunk_sizes = []
    avg_ttft = []
    avg_tpot = []
    # 只有在列表的才选择
    target_chunk_sizes = [128, 256, 512, 1024, 2048]

    for filename in os.listdir(log_dir):
        if filename.endswith('.json'):
            try:
                # 从文件名提取 chunk size
                chunk_size = int(filename.split('_')[1].replace('chunk_', '').replace('.json', ''))
                # 仅提取需要的 chunk size
                if chunk_size not in target_chunk_sizes:
                    continue
                chunk_sizes.append(chunk_size)

                ttft_values = []
                tpot_values = []

                with open(os.path.join(log_dir, filename), 'r') as f:
                    lines = f.readlines()[1:]  # 跳过第一条记录
                    for line in lines:
                        data = json.loads(line)
                        ttft_values.append(data['TTFT']['p50'] * 1000)  # 转换为 ms
                        tpot_values.append(data['TPOT']['p50'])

                # 检查列表是否为空
                if ttft_values and tpot_values:
                    # 计算平均值
                    avg_ttft.append(np.mean(ttft_values))
                    avg_tpot.append(np.mean(tpot_values))
                else:
                    print(f"Warning: No valid data in {log_dir}/{filename} after skipping the first line. Skipping this file.")
                    chunk_sizes.pop()  

            except (IndexError, ValueError, KeyError, json.JSONDecodeError):
                continue
    # 画图
    draw_ttft_tpot(chunk_sizes,avg_ttft,avg_tpot)
    # 保存
    save(log_dir)
    

def draw_ttft_tpot(chunk_sizes,avg_ttft,avg_tpot):
    # 对数据按 chunk size 排序
    if chunk_sizes:
        sorted_indices = np.argsort(chunk_sizes)
        chunk_sizes = np.array(chunk_sizes)[sorted_indices]
        avg_ttft = np.array(avg_ttft)[sorted_indices]
        avg_tpot = np.array(avg_tpot)[sorted_indices]

        # 使用序号作为 X 轴位置
        x_positions = np.arange(len(chunk_sizes))

        # 创建双 Y 轴线图（使用 constrained_layout 对齐）
        fig, ax1 = plt.subplots(figsize=(7, 3.0), constrained_layout=True)

        # 左轴：TTFT（ms）
        ax1.set_xlabel('Number of Batch Tokens')
        ax1.set_ylabel('Average TTFT (ms)')
        ax1.plot(x_positions, avg_ttft, color=COLOR_TTFT, linestyle='-', linewidth=2.0, alpha=0.98,
                 marker='o', markersize=4.0)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=len(x_positions)))

        # 右轴：TPOT（ms/token）
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average TPOT (ms/token)')
        ax2.plot(x_positions, avg_tpot, color=COLOR_TPOT, linestyle='-', linewidth=2.0, alpha=0.98,
                 marker='s', markersize=4.0)

        # 由于全局关闭了右侧脊线，这里对 twinx 的右脊线单独开启，保持可读
        if 'right' in ax2.spines:
            ax2.spines['right'].set_visible(True)

        # 仅 y 方向网格（左轴），轻量
        ax1.grid(True, axis='y', linestyle='-', linewidth=0.4, alpha=0.25)

        # 注释（简洁、深灰色）
        # left_text = "Optimized for Throughput (Low TPOT),\nbut high TTFT."
        # ax1.text(0.12, 0.06, left_text, transform=ax1.transAxes,
        #          fontsize=8, color='#444444', ha='left', va='bottom')

        right_text = "Optimized for Throughput (Low TTFT),\nbut creates head-of-line blocking."
        ax1.text(0.96, 0.50, right_text, transform=ax1.transAxes,
                 fontsize=8, color='#444444', ha='right', va='top')

        # 图例置于上方中间，横排
        custom_lines = [
            Line2D([0], [0], color=COLOR_TTFT, marker='o', markersize=4.0, lw=2.0, label='Average TTFT'),
            Line2D([0], [0], color=COLOR_TPOT, marker='s', markersize=4.0, lw=2.0, label='Average TPOT')
        ]
        ax1.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 1.18),
                   ncol=2, fontsize=10, frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.2)

        # X 轴刻度：显示 chunk 数字
        plt.xticks(x_positions, chunk_sizes)
        ax1.set_xlim([min(x_positions) - 0.2, max(x_positions) + 0.2])

    else:
        print("No valid data found in all files. Please check your log files.")


def save(log_dir):
        # 保存图表操作
    # 获取 log_dir 后两级目录名
    path_parts = os.path.normpath(log_dir).split(os.sep)
    if len(path_parts) >= 2:
        dir_name = '_'.join(path_parts[-2:])
    else:
        dir_name = 'ttft_tpot_plot'
        
    # svg_dir = "svg"
    # if not os.path.exists(svg_dir):
    #     os.makedirs(svg_dir, exist_ok=True)
    # plt.savefig(f'{svg_dir}/{dir_name}.svg', format='svg')

    plt.savefig('./ttft_tpot_plot.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.savefig('./ttft_tpot_plot.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.04)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default="./qps_0.5_prompt-1k")
    args = parser.parse_args()
    apply_topconf_style()
    data_process(args.log_dir)