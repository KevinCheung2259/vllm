import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.ticker import MaxNLocator


def apply_topconf_style():
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': [
            'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei', 'SimHei',
            'DejaVu Sans', 'Liberation Sans', 'Arial'
        ],
        'axes.unicode_minus': False,
        'axes.linewidth': 1.2,
        'axes.titlesize': 16,
        'axes.labelsize': 18,
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
        'grid.alpha': 0.25,
        'path.simplify': True,
        'path.simplify_threshold': 0.5,
        'agg.path.chunksize': 10000,
    })

# 生成模拟数据
def generate_simulation_data():
    # 使用字典存储数据
    data = {
        'categories': ['ShareGPT', 'Coder', 'FlowGPT-Q', 'FlowGPT-T'],
        'group1': {'name': 'SynergySched', 'values': [99.37, 73.65, 67.06, 61.85], 'color': 'red', 'hatch': ''},
        'group2': {'name': 'Latency-Based', 'values': [52.59, 31.04, 23.56, 27.12], 'color': '#999900', 'hatch': '/'},
        'group3': {'name': 'Weighted-Based', 'values': [84.28, 70.80, 58.13,36.26], 'color': '#009999', 'hatch': '\\'},
        'group4': {'name': 'Least-Loaded', 'values': [86.63, 66.06, 56.42, 46.08], 'color': '#FF66FF', 'hatch': 'x'}
    }
    return data

# 绘制多组柱状图
def plot_multi_group_bar(data):
    apply_topconf_style()
    # 获取数据
    categories = data['categories']
    groups = [data['group1'], data['group2'], data['group3'], data['group4']]
    n_categories = len(categories)
    n_groups = len(groups)
    
    # 设置柱状图宽度
    width = 0.8 / n_groups  # 总宽度为0.8，平均分配给各组
    
    # 创建图形和坐标轴（与 qps 小图一致大小）
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    
    # 为每个组别绘制柱状图
    x = np.arange(n_categories)
    for i, group in enumerate(groups):
        # 计算每个组的x位置偏移
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            group['values'],
            width,
            label=group['name'],
            color=group['color'],
            hatch=group['hatch'],
            edgecolor='black',
            linewidth=0.6,
        )
        
        # # 在柱子上方添加数值标签
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.annotate(f'{height}',
        #                 xy=(bar.get_x() + bar.get_width() / 2, height),
        #                 xytext=(0, 3),  # 3点垂直偏移
        #                 textcoords="offset points",
        #                 ha='center', va='bottom',
        #                 fontsize=10)
    
    # 设置坐标轴和标题
    # ax.set_xlabel('类别', fontsize=14)
    ax.set_ylabel('SLO Attainment (%)')
    # ax.set_title('不同方法在各类别上的比较', fontsize=16, fontweight='bold')
    
    # 设置x轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='center')
    ax.tick_params(axis='x', labelsize=12)
    
    # 设置y轴刻度数
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # 保持与统一样式一致的刻度样式
    
    # 添加图例
    ax.legend(loc='upper right', ncol=2, fontsize=10, frameon=False)
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.35)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    # 保存图像
    output_png = f"{output_dir}/heterogeneous_slo.png"
    output_pdf = f"{output_dir}/heterogeneous_slo.pdf"

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    
    plt.close(fig)

if __name__ == '__main__':
    # 生成模拟数据
    simulation_data = generate_simulation_data()
    
    # 绘制并保存柱状图
    plot_multi_group_bar(simulation_data)