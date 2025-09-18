#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于生成 "Model Run Time vs Total Schedule Tokens" 散点图
支持传入多个profiling文件夹，显示不同config下的数据对比
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
from typing import List, Dict, Optional, Tuple


class ModelRuntimeVsTokensPlotGenerator:
    """生成model runtime与total schedule tokens关系图的类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.setup_style()
    
    def setup_style(self):
        """设置matplotlib样式，与其他图表保持一致"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 23,
            'xtick.labelsize': 19,
            'ytick.labelsize': 19,
            'legend.fontsize': 14,
            'figure.figsize': (7, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.unicode_minus': False,
            # 简化路径以降低矢量复杂度
            'path.simplify': True,
            'path.simplify_threshold': 0.5,
            # 将超长路径分块渲染，避免PDF过慢
            'agg.path.chunksize': 10000,
            # 提高PDF压缩等级
            'pdf.compression': 9,
        })
    
    def read_profiling_data(self, log_file_or_dir: str, config_name: str = None) -> Optional[pd.DataFrame]:
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
                        # 添加配置标识
                        if config_name:
                            entry['config'] = config_name
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
            print(f"✅ 成功读取 {len(df)} 条profiling数据 ({config_name})")
        
        return df
    
    def calculate_total_schedule_tokens(self, df: pd.DataFrame) -> pd.Series:
        """计算total schedule tokens (chunk_sizes的总和)"""
        return df['chunk_sizes'].apply(
            lambda x: sum(x) if isinstance(x, list) else x
        )
    
    def generate_multi_config_plot(self, config_data_dict: Dict[str, pd.DataFrame], 
                                 save_path: str = None, title: str = None,
                                 rasterized: bool = False,
                                 max_points_per_config: Optional[int] = None,
                                 point_size: int = 50,
                                 alpha: float = 0.7,
                                 remove_edgecolors: bool = False) -> plt.Figure:
        """
        生成多配置的model runtime vs total schedule tokens散点图
        
        Args:
            config_data_dict: 配置名称到DataFrame的映射
            save_path: 图表保存路径（可选）
            title: 图表标题（可选）
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        # 定义颜色和标记样式
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        config_names = list(config_data_dict.keys())
        
        # 为每个配置绘制散点图
        for i, (config_name, df) in enumerate(config_data_dict.items()):
            if df is None or df.empty:
                continue
                
            # 计算total schedule tokens
            df_plot = df
            if max_points_per_config is not None and len(df_plot) > max_points_per_config:
                df_plot = df_plot.sample(n=max_points_per_config, random_state=42)
            total_tokens = self.calculate_total_schedule_tokens(df_plot)
            
            # 绘制散点图
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            edgecolor_value = 'none' if remove_edgecolors else 'white'
            linewidth_value = 0.0 if remove_edgecolors else 0.5
            ax.scatter(total_tokens, df_plot['model_run_duration_ms'], 
                      c=color, marker=marker, alpha=alpha, s=point_size, 
                      label=config_name, edgecolors=edgecolor_value, linewidth=linewidth_value,
                      rasterized=rasterized)
        
        # 设置图表属性
        ax.set_xlabel('Total Scheduled Tokens', fontsize=23, labelpad=10)
        ax.set_ylabel('Model Run Time (ms)', fontsize=23, labelpad=10)
        
        if title:
            ax.set_title(title, fontsize=18, pad=20)
        
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=2)
        ax.tick_params(axis='both', pad=8, labelsize=19)
        ax.legend(frameon=True, fancybox=True, shadow=True, 
                 loc='lower right', markerscale=1.8)
        
        # 设置坐标轴范围，使图表更紧凑
        ax.set_xlim(0, 4200)
        ax.set_ylim(0, 205)
        ax.set_xticks([0, 1024, 2048, 3072, 4096])
        ax.set_xticklabels(['0', '1024', '2048', '3072', '4096'])
        
        plt.subplots_adjust(left=0.12, right=0.85, bottom=0.15, top=0.92)
        
        # 保存图表
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"💾 图表已保存至: {save_path}")
        
        return fig
    
    def generate_single_config_plot(self, df: pd.DataFrame, config_name: str = "Config", 
                                   save_path: str = None, title: str = None, 
                                   color_by_batch_size: bool = True,
                                   rasterized: bool = False,
                                   max_points: Optional[int] = None,
                                   point_size: int = 60,
                                   alpha: float = 0.7,
                                   remove_edgecolors: bool = False) -> plt.Figure:
        """
        生成单配置的model runtime vs total schedule tokens散点图
        
        Args:
            df: 包含profiling数据的DataFrame
            config_name: 配置名称
            save_path: 图表保存路径（可选）
            title: 图表标题（可选）
            color_by_batch_size: 是否根据batch size着色
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        # 计算total schedule tokens
        df_plot = df
        if max_points is not None and len(df_plot) > max_points:
            df_plot = df_plot.sample(n=max_points, random_state=42)
        total_tokens = self.calculate_total_schedule_tokens(df_plot)
        
        if color_by_batch_size:
            # 计算batch size（chunk_sizes的长度）
            batch_sizes = []
            for chunk_sizes in df_plot['chunk_sizes']:
                if isinstance(chunk_sizes, list):
                    batch_sizes.append(len(chunk_sizes))
                else:
                    batch_sizes.append(1)
            
            batch_sizes = np.array(batch_sizes)
            
            # 使用batch size作为颜色映射
            edgecolor_value = 'none' if remove_edgecolors else 'white'
            linewidth_value = 0.0 if remove_edgecolors else 0.5
            scatter = ax.scatter(total_tokens, df_plot['model_run_duration_ms'], 
                               c=batch_sizes, cmap='viridis', alpha=alpha, s=point_size, 
                               edgecolors=edgecolor_value, linewidth=linewidth_value,
                               rasterized=rasterized)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Batch Size', fontsize=16, labelpad=5)
            cbar.ax.tick_params(labelsize=14)
        else:
            # 单色散点图
            edgecolor_value = 'none' if remove_edgecolors else 'white'
            linewidth_value = 0.0 if remove_edgecolors else 0.5
            ax.scatter(total_tokens, df_plot['model_run_duration_ms'], 
                      c='#1f77b4', alpha=alpha, s=point_size, 
                      edgecolors=edgecolor_value, linewidth=linewidth_value, label=config_name,
                      rasterized=rasterized)
            # 放在右下角
            ax.legend(loc='lower right')
        
        # 设置图表属性
        ax.set_xlabel('Total Scheduled Tokens', fontsize=23, labelpad=10)
        ax.set_ylabel('Model Run Time (ms)', fontsize=23, labelpad=10)
        
        if title:
            ax.set_title(title, fontsize=18, pad=20)
        
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=2)
        ax.tick_params(axis='both', pad=8, labelsize=19)
        
        # 设置坐标轴范围
        # x轴为256, 512, 1024, 2048, 4096
        ax.set_xlim(256, 4096)
        ax.set_ylim(bottom=0)
        
        plt.subplots_adjust(left=0.12, right=0.85, bottom=0.15, top=0.92)
        
        # 保存图表
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"💾 图表已保存至: {save_path}")
        
        return fig


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成Model Run Time vs Total Schedule Tokens散点图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python draw_runtime_vs_tokens.py config1:/path/to/profiling1 config2:/path/to/profiling2
  python draw_runtime_vs_tokens.py --single /path/to/profiling --output runtime_vs_tokens.png
        """
    )
    
    parser.add_argument('configs', nargs='*', 
                       help='配置格式: config_name:/path/to/profiling_dir')
    parser.add_argument('--single', type=str,
                       help='单配置模式：指定单个profiling目录路径')
    parser.add_argument('--output', '-o', type=str, 
                       default='model_runtime_vs_tokens.png',
                       help='输出图片文件名 (默认: model_runtime_vs_tokens.png)')
    parser.add_argument('--title', type=str,
                       help='图表标题')
    parser.add_argument('--color-by-batch', action='store_true',
                       help='在单配置模式下根据batch size着色')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--rasterized', action='store_true',
                       help='将散点以栅格方式嵌入PDF，显著降低PDF渲染开销')
    parser.add_argument('--max-points-per-config', type=int, default=None,
                       help='每个配置最多绘制的点数，超出将随机下采样')
    parser.add_argument('--point-size', type=int, default=50,
                       help='散点大小')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='散点透明度')
    parser.add_argument('--no-edges', action='store_true',
                       help='移除散点边框以减少矢量路径复杂度')
    
    args = parser.parse_args()
    
    # 创建绘图器
    plotter = ModelRuntimeVsTokensPlotGenerator(verbose=args.verbose)
    
    if args.single:
        # 单配置模式
        df = plotter.read_profiling_data(args.single, "Single Config")
        if df is None or df.empty:
            print("❌ 无法读取数据或数据为空")
            return
        
        fig = plotter.generate_single_config_plot(
            df, 
            config_name="Config",
            save_path=args.output,
            title=args.title,
            color_by_batch_size=args.color_by_batch,
            rasterized=args.rasterized,
            max_points=args.max_points_per_config,
            point_size=args.point_size,
            alpha=args.alpha,
            remove_edgecolors=args.no_edges,
        )
        
    else:
        # 多配置模式
        if not args.configs:
            print("❌ 请提供配置数据")
            print("使用 --help 查看帮助")
            return
        
        config_data_dict = {}
        
        for config_spec in args.configs:
            try:
                if ':' in config_spec:
                    config_name, config_path = config_spec.split(':', 1)
                else:
                    config_name = Path(config_spec).name
                    config_path = config_spec
                
                df = plotter.read_profiling_data(config_path, config_name)
                if df is not None and not df.empty:
                    config_data_dict[config_name] = df
                    
            except Exception as e:
                print(f"❌ 处理配置 {config_spec} 时出错: {e}")
                continue
        
        if not config_data_dict:
            print("❌ 没有成功读取任何配置数据")
            return
        
        fig = plotter.generate_multi_config_plot(
            config_data_dict,
            save_path=args.output,
            title=args.title,
            rasterized=args.rasterized,
            max_points_per_config=args.max_points_per_config,
            point_size=args.point_size,
            alpha=args.alpha,
            remove_edgecolors=args.no_edges,
        )
    
    # 显示图表
    plt.show()


if __name__ == '__main__':
    main() 