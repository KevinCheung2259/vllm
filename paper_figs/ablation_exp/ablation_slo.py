import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取数据
def get_data():
    # 使用字典存储数据
    data = {
        'categories': ['Summarization', 'FlowGPT-Q', 'FlowGPT-T','ShareGPT','Coding'],
        'group1': {'name': 'SynergySched', 'values': [96.77, 96.29, 80.47,96.12, 79.69,], 'color': 'red', 'hatch': ''},
        'group2': {'name': 'Sarathi PRISM', 'values': [39.88, 28.84, 64.11, 88.93,61.18,], 'color': 'brown', 'hatch': '/'},
        'group3': {'name': 'LENS RR', 'values': [88.52, 75.88, 69.43,86.10,73.57,], 'color': 'orange', 'hatch': '\\'},
        'group4': {'name': 'Sarathi RR', 'values': [34.19, 14.16, 29.09 ,79.56,50.90], 'color': '#CCCC00', 'hatch': '-'}
    }
    return data

# 绘制多组柱状图
def plot_multi_group_bar(data):
    # 获取数据
    categories = data['categories']
    groups = [data['group1'], data['group2'], data['group3'], data['group4']]
    n_categories = len(categories)
    n_groups = len(groups)
    
    # 设置柱状图宽度
    width = 0.8 / n_groups  # 总宽度为0.8，平均分配给各组
    
    default_fontsize = 12
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    
    # 为每个组别绘制柱状图
    x = np.arange(n_categories)
    for i, group in enumerate(groups):
        # 计算每个组的x位置偏移
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, group['values'], width, 
                      label=group['name'], 
                      color=group['color'],
                      hatch=group['hatch'],
                      edgecolor='black',
                      linewidth=0.7)
        
    # 设置坐标轴和标题
    # ax.set_xlabel('类别', fontsize=14)
    ax.set_ylabel('SLO Attainment (%)', fontsize=default_fontsize)
    # ax.set_title('不同方法在各类别上的比较', fontsize=16, fontweight='bold')
    
    # 设置x轴刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=default_fontsize)
    
    # 设置y轴刻度数
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='y', labelsize=12)
    
    # 添加图例
    ax.legend(fontsize=default_fontsize, 
             loc='lower center', 
             bbox_to_anchor=(0.5, 1),
             ncol=4,
             columnspacing=1.0,
             frameon=False)
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 增加顶部空间

    # # 添加图例
    # ax.legend(fontsize=default_fontsize, loc='best')
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 获取当前文件的完整路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 保存图像
    output_png = f"{current_dir}/e2e_ablation.png"
    output_pdf = f"{current_dir}/e2e_ablation.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    
    # 显示图形
    plt.show()

if __name__ == '__main__':
    # 获取数据
    data = get_data()
    
    # 绘制并保存柱状图
    plot_multi_group_bar(data)