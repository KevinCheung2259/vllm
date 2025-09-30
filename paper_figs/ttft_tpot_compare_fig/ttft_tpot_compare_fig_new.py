import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import argparse
import datetime


def data_process(log_dir):
    chunk_sizes = []
    avg_ttft = []
    avg_tpot = []
    target_chunk_sizes = [128, 256, 512, 1024, 2048]

    for filename in os.listdir(log_dir):
        if filename.endswith('.json'):
            try:
                chunk_size = int(filename.split('_')[1].replace('chunk_', '').replace('.json', ''))
                if chunk_size not in target_chunk_sizes:
                    continue
                chunk_sizes.append(chunk_size)

                ttft_values = []
                tpot_values = []

                with open(os.path.join(log_dir, filename), 'r') as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        data = json.loads(line)
                        ttft_values.append(data['TTFT']['p50'] * 1000)
                        tpot_values.append(data['TPOT']['p50'])

                if ttft_values and tpot_values:
                    avg_ttft.append(np.mean(ttft_values))
                    avg_tpot.append(np.mean(tpot_values))
                else:
                    print(f"Warning: No valid data in {log_dir}/{filename}")
                    chunk_sizes.pop()

            except (IndexError, ValueError, KeyError, json.JSONDecodeError):
                continue
    return {'chunk_sizes': chunk_sizes, 'avg_ttft': avg_ttft, 'avg_tpot': avg_tpot}


def simulator(log_dir):
    length = 25  # 生成30个数据点
    avg_ttft = []
    avg_tpot = []
    
    # y = 1500 -25x
    # 两个已知点的坐标
    point1 = (10, 1250)
    point2 = (55, 125)
    
    # 计算直线方程 y = mx + b
    x1, y1 = point1
    x2, y2 = point2
    m = (y2 - y1) / (x2 - x1)  # 斜率
    b = y1 - m * x1  # 截距
    
    # 生成 length 个均匀分布在x1和x2之间的TPOT值
    avg_tpot = np.linspace(x1, x2, length)
    
    # 根据直线方程计算对应的TTFT值，并添加随机噪声让数据分布在直线周边
    noise_level = 150  # 噪声水平，可以根据需要调整
    for x in avg_tpot:
        # 计算直线上的y值
        y_line = m * x + b
        # 添加随机噪声
        noise = np.random.normal(0, noise_level)  # 高斯分布噪声
        y = y_line + noise
        avg_ttft.append(y)
    
    avg_ttft = np.array(avg_ttft)
    avg_tpot = np.array(avg_tpot)

    chunk_sizes = [0]*len(avg_ttft)

    return {'chunk_sizes': chunk_sizes, 'avg_ttft': avg_ttft, 'avg_tpot': avg_tpot}

def save_simulation_data(data,current_dir = "." ):
    """将模拟数据以追加模式写入JSON文件，包含时间戳"""
    # 获取当前时间，格式为ISO 8601
    current_time = datetime.datetime.now().isoformat()
    
    output_file=f"{current_dir}/data.jsonl"
    # 准备要写入的数据，添加时间戳
    data_to_save = {
        "timestamp": current_time,
        "avg_ttft": data["avg_ttft"].tolist() if isinstance(data["avg_ttft"], np.ndarray) else data["avg_ttft"],
        "avg_tpot": data["avg_tpot"].tolist() if isinstance(data["avg_tpot"], np.ndarray) else data["avg_tpot"]
    }
    
    # 以追加模式打开文件并写入数据
    with open(output_file, 'a') as f:
        json.dump(data_to_save, f)
        f.write('\n')  # 确保每条数据占一行

def draw_ttft_tpot(data,metrics):

    # 初始化数据
    ttft_slo = metrics['ttft_slo']

    chunk_sizes = data['chunk_sizes']
    avg_ttft = data['avg_ttft']
    avg_tpot = data['avg_tpot']



    if chunk_sizes:
        sorted_indices = np.argsort(chunk_sizes)
        chunk_sizes = np.array(chunk_sizes)[sorted_indices]
        avg_ttft = np.array(avg_ttft)[sorted_indices]
        avg_tpot = np.array(avg_tpot)[sorted_indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 调整坐标轴刻度,限制刻度数量
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        # 核心绘图要素
        ax.set_xlabel('Average TPOT (ms)', labelpad=15,fontsize=metrics['font_size'])
        ax.set_ylabel('Average TTFT (ms)', labelpad=15,fontsize=metrics['font_size'])
        ax.plot(avg_tpot, avg_ttft, 'o', color='blue', linewidth=4, markersize=11,alpha=1)
        
         # 添加线性拟合的虚线
        if len(avg_tpot) > 1 and len(avg_ttft) > 1:
            # 执行线性回归拟合
            coefficients = np.polyfit(avg_tpot, avg_ttft, 1)  # 1表示一次多项式(直线)
            polynomial = np.poly1d(coefficients)  # 创建拟合函数
            
            # 生成拟合直线的x值范围
            x_fit = np.linspace(min(avg_tpot), max(avg_tpot), 100)
            y_fit = polynomial(x_fit)  # 计算对应的y值
            
            # 绘制拟合的虚线
            ax.plot(x_fit, y_fit, '--', color='red', linewidth=6, label='Fitted Line')

        # 坐标轴刻度参数
        ax.tick_params(axis='both', 
               labelsize=metrics['font_size'],  # 刻度字体大小
               width=1,               # 刻度线粗细
               length=4)              # 刻度线长度
        

        # 添加垂直参考线

        ax.axvline(x=metrics['tpot_slo'], color='orange',
                   linestyle='--', linewidth=4,label='TPOT Threshold')
        
        # 在垂直参考线旁边添加水平文字"TPOT SLO"
        ax.text(metrics['tpot_slo']+0.5, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05-140, 'TPOT SLO', 
                verticalalignment='bottom', horizontalalignment='left',
                color='black', fontsize=metrics['font_size_slo'],fontweight='bold')

        # 添加水平参考线        
        ax.axhline(y=metrics['ttft_slo'], color='orange',linestyle='--',
                    linewidth=4,label='TTFT Threshold')
        
        # 在水平参考线旁边添加水平文字"TTFT SLO"
        ax.text(ax.get_xlim()[0]-2, metrics['ttft_slo'], 'TTFT SLO',verticalalignment='bottom',
                 horizontalalignment='left',color='black', fontsize=metrics['font_size_slo'],fontweight='bold')
        
        # 坐标范围自适应
        x_padding = np.ptp(avg_tpot)*0.1
        y_padding = np.ptp(avg_ttft)*0.1
        ax.set_xlim(np.min(avg_tpot)-x_padding, np.max(avg_tpot)+x_padding)
        ax.set_ylim(np.min(avg_ttft)-y_padding, np.max(avg_ttft)+y_padding)

        # 自定义坐标刻度
        # custom_xticks = [20, 35, 50, 65]  # 自定义的x轴刻度值
        # ax.set_xticks(custom_xticks)
        # custom_yticks = [200, 500, 800, 1100,1400]  # 自定义的y轴刻度值
        # ax.set_yticks(custom_yticks)

        # 添加阴影区域：
        # 1. 从x1到x2，所有位于拟合线下方的区域
        if len(avg_tpot) > 1 and len(avg_ttft) > 1:
            
            # 生成密集的x值
            x_shadow = np.linspace(metrics['shadow_x'][0], metrics['shadow_x'][1], 1000)
            # 计算拟合线对应的y值
            y_shadow_fit = polynomial(x_shadow)
            # 获取y轴最小值作为阴影区域的下边界
            y_bottom = ax.get_ylim()[0]
            
            # 绘制拟合线下方的阴影区域（透明度0.5）
            ax.fill_between(x_shadow, y_bottom, y_shadow_fit, color='green', alpha=0.5, hatch='/',
                           edgecolor='black',linewidth =2,label='Below Fitted Line')
        
        # 2. 从y1到y2，所有位于拟合线左侧的区域
        if len(avg_tpot) > 1 and len(avg_ttft) > 1:

            # 生成密集的y值
            y_shadow = np.linspace(metrics['shadow_y'][0], metrics['shadow_y'][1], 1000)
            # 计算拟合线的反函数，得到对应的x值
            # y = mx + b => x = (y - b)/m
            m, b = coefficients
            if m != 0:  # 避免除以零
                x_shadow_fit = (y_shadow - b) / m
                # 获取x轴最小值作为阴影区域的左边界
                x_left = ax.get_xlim()[0]
                
                # 绘制拟合线左侧的阴影区域（透明度0.5）
                ax.fill_betweenx(y_shadow, x_left, x_shadow_fit, color='green', alpha=0.5, hatch='\\',
                                edgecolor='black',linewidth =2,label='Left of Fitted Line')
                

        # 设置坐标轴边界显示
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
        # 添加坐标轴箭头
        # # 设置箭头样式
        # ax.spines['bottom'].set_position(('data', ax.get_ylim()[0]))
        # ax.spines['left'].set_position(('data', ax.get_xlim()[0]))
        
        # # 添加箭头
        # ax.plot(1, 0, '>', transform=ax.transAxes, clip_on=False, color='k', markersize=10)
        # ax.plot(0, 1, '^', transform=ax.transAxes, clip_on=False, color='k', markersize=10)

        plt.grid(True,alpha=0.5)
        plt.tight_layout()


def save(log_dir):

    parent_dir = os.path.dirname(log_dir)

    dir_name = "ttft_tpot_plot_new"
    
    plt.savefig(f'{parent_dir}/{dir_name}.png', format='png', dpi=300)
    plt.savefig(f'{parent_dir}/{dir_name}.pdf', format='pdf', dpi=300)
    plt.close()

if __name__ == "__main__":

    # 获取当前文件的完整路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    parser = argparse.ArgumentParser(description='Generate TTFT vs TPOT analysis plots')
    parser.add_argument('--log-dir', 
                        type=str, 
                        default=f"{current_dir}/qps_0.5_prompt-1k",
                        help='Directory containing JSON log files')
    
    args = parser.parse_args()
    
    # data = data_process(args.log_dir)
    data = simulator(args.log_dir)
    save_simulation_data(data,current_dir)
    metrics = {
        'font_size': 24,
        'font_size_slo': 20,
        'tpot_slo': 45,
        'ttft_slo': 900,
        'shadow_x':[24,45],
        'shadow_y':[375,900],
        }
    draw_ttft_tpot(data, metrics)
    save(args.log_dir)
