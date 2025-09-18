#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于生成 "Chunk Size vs Model Run Time" 散点图
实现从数据读取到图表生成的端到端功能
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path


class ChunkRuntimePlotGenerator:
    """生成chunk size与运行时间关系图的类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        # 添加与热力图相同的样式设置
        self.setup_style()
    
    def setup_style(self):
        """设置matplotlib样式，与热力图保持一致"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 23,
            'xtick.labelsize': 19,
            'ytick.labelsize': 19,
            'legend.fontsize': 16,
            'figure.figsize': (7, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.unicode_minus': False,
        })
    
    def read_profiling_data(self, log_file_or_dir: str) -> pd.DataFrame:
        """读取profiling数据"""
        log_path = Path(log_file_or_dir)
        
        if log_path.is_dir():
            jsonl_files = list(log_path.glob('*.jsonl'))
            if not jsonl_files:
                print(f"❌ 在目录 {log_file_or_dir} 中没有找到jsonl文件")
                return None
            log_files = jsonl_files
            if self.verbose:
                print(f"📁 找到目录: {log_file_or_dir}")
                print(f"📄 使用文件: {len(jsonl_files)} 个")
        else:
            if not log_path.exists():
                print(f"❌ 日志文件 {log_path} 不存在")
                return None
            log_files = [log_path]
            if self.verbose:
                print(f"📄 使用单个文件: {log_path}")
        
        # 读取数据
        data = []
        batch_id_offset = 0
        
        # 排序文件
        if len(log_files) > 1:
            try:
                log_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            except Exception:
                log_files.sort(key=lambda x: x.name)
        
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 去掉前后10行
                for line in lines[10:-10]:
                    try:
                        entry = json.loads(line.strip())
                        if 'batch_id' in entry:
                            entry['batch_id'] += batch_id_offset
                        data.append(entry)
                    except json.JSONDecodeError:
                        continue
            batch_id_offset += len(lines[10:-10])
        
        if not data:
            return None
        
        # 数据清洗
        data = [item for item in data if item.get('schedule_duration_ms', 0) < 300]
        data = [item for item in data if item.get('model_run_duration_ms', 0) < 200]
        
        df = pd.DataFrame(data)
        
        if self.verbose:
            print(f"✅ 成功读取 {len(df)} 条profiling数据")
        
        return df
    
    def generate_chunk_runtime_plot(self, df: pd.DataFrame, save_path: str = None, 
                                   title: str = None) -> plt.Figure:
        """
        生成chunk size vs runtime散点图
        
        Args:
            df: 包含profiling数据的DataFrame
            save_path: 图表保存路径（可选）
            title: 图表标题（可选）
            
        Returns:
            matplotlib Figure对象
        """
        # 创建图表，参考draw.py的风格
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # 处理chunk_sizes数据 - 计算每个批次的总chunk size
        prefill_chunk_totals = df['chunk_sizes'].apply(
            lambda x: sum(x) if isinstance(x, list) else x
        )
        
        # 计算batch size（chunk_sizes的长度），batch size越大颜色越深
        batch_sizes = []
        for chunk_sizes in df['chunk_sizes']:
            if isinstance(chunk_sizes, list):
                batch_sizes.append(len(chunk_sizes))
            else:
                batch_sizes.append(1)  # 如果不是列表，默认长度为1
        
        # 转换为numpy数组便于处理
        batch_sizes = np.array(batch_sizes)
        
        # 绘制散点图，使用batch size作为颜色映射
        # 使用coolwarm颜色映射：蓝色(小batch size)到红色(大batch size)
        scatter = ax.scatter(prefill_chunk_totals, df['model_run_duration_ms'], 
                           c=batch_sizes, cmap='coolwarm', alpha=0.8, s=65, 
                           edgecolors='white', linewidth=0.6)
        
        # 设置图表属性，参考draw.py的风格
        ax.set_xlabel('Total Scheduled Tokens', fontsize=23, labelpad=10)
        ax.set_ylabel('Model Run Time (ms)', fontsize=23, labelpad=10)
        
        if title:
            pass
            # ax.set_title(title, fontsize=18, pad=20)
        
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=2)
        
        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', pad=8, labelsize=19)
        
        # 添加颜色条说明batch size（显示实际的batch size范围）
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Batch Size', fontsize=16, labelpad=5)
        cbar.ax.tick_params(labelsize=14)
        
        # 设置颜色条刻度为整数（batch size通常是整数）
        if len(np.unique(batch_sizes)) <= 10:  # 如果batch size种类不多，显示所有值
            cbar.set_ticks(np.unique(batch_sizes))
        else:  # 如果种类很多，显示等间距的整数刻度
            min_batch = int(batch_sizes.min())
            max_batch = int(batch_sizes.max())
            tick_values = np.linspace(min_batch, max_batch, 8, dtype=int)
            cbar.set_ticks(tick_values)
        
        # 调整布局，与热力图保持一致
        plt.subplots_adjust(left=0.12, right=0.85, bottom=0.15, top=0.92)
        
        # 保存图表，与热力图保持一致的参数
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            if self.verbose:
                print(f"📊 图表已保存至: {save_path}")
            
            # 同时保存PDF格式
            pdf_path = save_path.replace(".png", ".pdf")
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf',
                       facecolor='white', edgecolor='none')
            if self.verbose:
                print(f"📊 PDF格式已保存至: {pdf_path}")
        
        return fig
    
    def run_end_to_end(self, log_path: str, save_path: str = None, 
                      title: str = None) -> bool:
        """
        运行端到端的流程：读取数据、生成图表
        
        Args:
            log_path: 日志文件或目录路径
            save_path: 图表保存路径（可选）
            title: 图表标题（可选）
            
        Returns:
            是否成功执行
        """
        try:
            # 1. 读取数据
            if self.verbose:
                print("📥 正在读取profiling数据...")
            
            df = self.read_profiling_data(log_path)
            if df is None or df.empty:
                print("❌ 未找到有效数据")
                return False
            
            # 检查必要的列
            if 'chunk_sizes' not in df.columns or 'model_run_duration_ms' not in df.columns:
                print("❌ 数据中缺少必要的列: chunk_sizes 或 model_run_duration_ms")
                return False
            
            # 2. 生成图表
            if self.verbose:
                print("📊 正在生成Chunk Size vs Runtime图表...")
            
            self.generate_chunk_runtime_plot(df, save_path=save_path, title=title)
            
            return True
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成Chunk Size vs Model Run Time散点图')
    parser.add_argument('log_path', type=str, nargs='?', 
                       default='profiling_result',
                       help='profiling数据文件或目录路径 (默认: profiling_result)')
    parser.add_argument('--save-path', type=str, 
                       default="./paper_figs/chunk_size_vs_runtime_fig/chunk_size_vs_runtime.png",
                       help='图表保存路径')
    parser.add_argument('--title', type=str,
                       help='图表标题')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    generator = ChunkRuntimePlotGenerator(verbose=args.verbose)
    
    success = generator.run_end_to_end(
        log_path=args.log_path,
        save_path=args.save_path,
        title=args.title
    )
    
    if success:
        print("✅ 图表生成完成！")
    else:
        print("❌ 图表生成失败！")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 