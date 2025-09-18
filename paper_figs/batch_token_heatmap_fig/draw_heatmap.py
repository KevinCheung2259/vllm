#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Size vs Total Tokens 热力图绘制脚本
参考draw_chunk_runtime样式，用于论文投稿

绘制批次大小(Batch Size)与总token数(Total Tokens)对模型运行时间的影响
支持数据缺失时的模型拟合填充

使用方法:
python draw_heatmap.py <数据文件或目录路径>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import argparse
import traceback

class BatchTokenHeatmapGenerator:
    """生成Batch Size vs Total Tokens热力图的类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        # 参考draw_chunk_runtime的样式设置
        self.setup_style()
    
    def setup_style(self):
        """设置matplotlib样式，参考draw_chunk_runtime"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 23,  # 参考draw_chunk_runtime
            'xtick.labelsize': 19, # 参考draw_chunk_runtime
            'ytick.labelsize': 19, # 参考draw_chunk_runtime
            'legend.fontsize': 16,
            'figure.figsize': (7, 6),  # 参考draw_chunk_runtime
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.unicode_minus': False,
        })
    
    def read_profiling_data(self, log_file_or_dir: str) -> pd.DataFrame:
        """读取profiling数据，参考draw_chunk_runtime的数据处理方式"""
        log_path = Path(log_file_or_dir)
        
        if log_path.is_dir():
            jsonl_files = list(log_path.glob('*.jsonl'))
            if not jsonl_files:
                if self.verbose:
                    print(f"❌ 在目录 {log_file_or_dir} 中没有找到jsonl文件")
                return None
            log_files = jsonl_files
            if self.verbose:
                print(f"📁 找到目录: {log_file_or_dir}")
                print(f"📄 使用文件: {len(jsonl_files)} 个")
        else:
            if not log_path.exists():
                if self.verbose:
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
                # 跳过前后10行以避免不稳定数据
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
            if self.verbose:
                print("❌ 没有找到有效的profiling数据")
            return None
        
        df = pd.DataFrame(data)
        
        # 数据清洗：移除异常值
        if self.verbose:
            print(f"📊 原始数据点数: {len(df)}")
        
        df = df[df.get('schedule_duration_ms', 0) < 300]
        df = df[df.get('model_run_duration_ms', 0) < 200]
        
        if self.verbose:
            print(f"✅ 清洗后数据点数: {len(df)}")
            print(f"✅ 成功读取 {len(df)} 条profiling数据")
        
        return df
    
    def extract_batch_and_token_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取batch size和total tokens特征"""
        def get_batch_size(sizes):
            if isinstance(sizes, list):
                return len(sizes)
            return np.nan
        
        def get_total_tokens(sizes):
            if isinstance(sizes, list):
                return sum(sizes)
            if isinstance(sizes, (int, float)):
                return float(sizes)
            return np.nan
        
        df['batch_size'] = df['chunk_sizes'].apply(get_batch_size)
        df['total_tokens'] = df['chunk_sizes'].apply(get_total_tokens)
        
        # 过滤有效数据
        valid_df = df[
            (df['batch_size'] >= 1) & (df['batch_size'] <= 120) &
            (df['total_tokens'] >= 1) & (df['total_tokens'] <= 4096) &
            (df['model_run_duration_ms'].notna())
        ].copy()
        
        if self.verbose:
            print(f"📈 提取特征后有效数据点数: {len(valid_df)}")
            print(f"📊 Batch Size 范围: {valid_df['batch_size'].min():.0f} - {valid_df['batch_size'].max():.0f}")
            print(f"📊 Total Tokens 范围: {valid_df['total_tokens'].min():.0f} - {valid_df['total_tokens'].max():.0f}")
            print(f"📊 Model Runtime 范围: {valid_df['model_run_duration_ms'].min():.2f} - {valid_df['model_run_duration_ms'].max():.2f} ms")
        
        return valid_df
    
    def runtime_model_function(self, batch_size, total_tokens, params):
        """
        运行时间模型函数：T(B,S) = T0 + αS + γ⋅(1-e^(-B/B0))
        - T0: 基础时间
        - αS: 对token数S的线性增长
        - γ⋅(1-e^(-B/B0)): 对batch size B的饱和增长函数
        """
        # 基础时间
        T0 = params['base_time']
        
        # 线性增长项 (total_tokens)
        linear_term = params['alpha'] * total_tokens
        
        # 饱和增长项 (batch_size) - 先快速增长后趋平
        saturation_term = params['gamma'] * (1 - np.exp(-batch_size / params['B0']))
        
        return T0 + linear_term + saturation_term
    
    def fit_model_to_data(self, valid_df: pd.DataFrame):
        """拟合模型参数到实际数据"""
        
        # 拟合前先剔除掉[batchsize在24-96且token数在1000-3840]的数据
        # 使用NOT条件来剔除同时满足两个条件的数据
        filtered_df = valid_df
        
        # 从过滤后的数据获取数组
        batch_sizes = filtered_df['batch_size'].values
        total_tokens = filtered_df['total_tokens'].values
        runtimes = filtered_df['model_run_duration_ms'].values
        
        if self.verbose:
            print(f"📊 拟合数据点数: {len(filtered_df)} (原始: {len(valid_df)})")
        
        def objective(params_vec):
            params = {
                'base_time': params_vec[0],    # T0
                'alpha': params_vec[1],        # α (线性系数)
                'gamma': params_vec[2],        # γ (饱和最大值)
                'B0': params_vec[3]            # B0 (饱和参数)
            }
            predicted = self.runtime_model_function(batch_sizes, total_tokens, params)
            return np.mean((predicted - runtimes) ** 2)
        
        # 调整初始参数和边界以适应新的饱和模型
        # [base_time(T0), alpha(α), gamma(γ), B0]
        initial_guess = [10.0, 0.01, 30.0, 20.0]
        bounds = [(1, 50), (0.001, 0.1), (5, 100), (5, 100)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        fitted_params = {
            'base_time': result.x[0],  # T0
            'alpha': result.x[1],      # α
            'gamma': result.x[2],      # γ
            'B0': result.x[3]          # B0
        }
        
        # 计算R²
        predicted = self.runtime_model_function(batch_sizes, total_tokens, fitted_params)
        ss_res = np.sum((runtimes - predicted) ** 2)
        ss_tot = np.sum((runtimes - np.mean(runtimes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        if self.verbose:
            print(f"📐 模型拟合完成:")
            print(f"   基础时间 (T0): {fitted_params['base_time']:.2f} ms")
            print(f"   线性系数 (α): {fitted_params['alpha']:.6f}")
            print(f"   饱和最大值 (γ): {fitted_params['gamma']:.2f}")
            print(f"   饱和参数 (B0): {fitted_params['B0']:.2f}")
            print(f"   R² = {r_squared:.4f}")
            
            # 验证模型行为 - 饱和函数应该表现为batch size增长时边际效应递减
            test_batch_sizes = [1, 12, 30, 60, 100]
            test_tokens = [100, 1000, 3500]
            print(f"📊 模型验证 (饱和函数验证) - 注意：以下为原始预测值，热力图中会限制并平滑处理:")
            for tokens in test_tokens:
                print(f"   Token={tokens}:")
                prev_runtime = 0
                for batch_size in test_batch_sizes:
                    predicted = self.runtime_model_function(batch_size, tokens, fitted_params)
                    if prev_runtime > 0:
                        increment = predicted - prev_runtime
                        print(f"     Batch={batch_size}: {predicted:.2f}ms (+{increment:.2f})")
                    else:
                        print(f"     Batch={batch_size}: {predicted:.2f}ms")
                    prev_runtime = predicted
        
        return fitted_params, r_squared
    
    def create_heatmap_data(self, valid_df: pd.DataFrame, resolution=(25, 25)):
        """创建热力图数据，智能处理缺失数据"""
        
        # 基于实际数据确定范围
        batch_min = max(1, valid_df['batch_size'].min())
        batch_max = min(120, valid_df['batch_size'].max())
        token_min = max(1, valid_df['total_tokens'].min())
        token_max = min(4096, valid_df['total_tokens'].quantile(0.99))
        
        if self.verbose:
            print(f'📊 数据范围: batch_size [{batch_min:.0f}, {batch_max:.0f}], total_tokens [{token_min:.0f}, {token_max:.0f}]')
        
        # 拟合模型到现有数据
        fitted_params, r_squared = self.fit_model_to_data(valid_df)
        
        # 创建网格
        batch_bins, token_bins = resolution
        batch_grid = np.linspace(batch_min, batch_max, batch_bins)
        token_grid = np.linspace(token_min, token_max, token_bins)
        
        # 初始化运行时间矩阵
        runtime_matrix = np.full((token_bins, batch_bins), np.nan)
        
        # 将实际数据映射到网格 - 创建数据密度更高的区域
        for _, row in valid_df.iterrows():
            batch_idx = np.argmin(np.abs(batch_grid - row['batch_size']))
            token_idx = np.argmin(np.abs(token_grid - row['total_tokens']))
            
            if np.isnan(runtime_matrix[token_idx, batch_idx]):
                runtime_matrix[token_idx, batch_idx] = row['model_run_duration_ms']
            else:
                # 如果已有数据，用原来的值与模型预测值平均
                runtime_matrix[token_idx, batch_idx] = (
                    runtime_matrix[token_idx, batch_idx] + row['model_run_duration_ms']) / 2
        
        # 统计数据覆盖率
        data_coverage = np.sum(~np.isnan(runtime_matrix)) / runtime_matrix.size
        if self.verbose:
            print(f'📊 数据覆盖率: {data_coverage:.1%}')
        
        # 使用模型填充缺失数据（使用原始预测值）
        batch_mesh, token_mesh = np.meshgrid(batch_grid, token_grid)
        for i in range(token_bins):
            for j in range(batch_bins):
                if np.isnan(runtime_matrix[i, j]):
                    predicted_runtime = self.runtime_model_function(
                        batch_mesh[i, j], token_mesh[i, j], fitted_params
                    )
                    # 直接使用原始预测值，不进行限制处理
                    runtime_matrix[i, j] = predicted_runtime
        
        # 不进行平滑处理，保持原始预测值
        # runtime_matrix = gaussian_filter(runtime_matrix, sigma=0.8)
        
        if self.verbose:
            print(f"📊 热力图网格大小: {runtime_matrix.shape}")
            print(f"📊 运行时间范围 (原始模型预测值): {runtime_matrix.min():.2f} - {runtime_matrix.max():.2f} ms")
        
        return runtime_matrix, batch_grid, token_grid, fitted_params, r_squared
    
    def plot_heatmap(self, runtime_matrix, batch_grid, token_grid, fitted_params, r_squared, 
                    save_path: str = None) -> plt.Figure:
        """绘制热力图，参考draw_chunk_runtime样式"""
        
        # 创建图表，参考draw_chunk_runtime的尺寸
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        # 使用viridis颜色映射
        cmap = plt.cm.viridis
        
        # 绘制热力图，关键：修改extent参数确保坐标轴显示实际数值而非范围
        im = ax.imshow(
            runtime_matrix,
            extent=[batch_grid[0], batch_grid[-1], token_grid[0], token_grid[-1]],
            origin='lower',
            aspect='auto',
            cmap=cmap,
            interpolation='bilinear'
        )
        
        # 基于拟合模型生成理论等高线
        # 1. 创建高分辨率网格用于理论模型预测
        model_resolution = (100, 100)  # 高分辨率确保平滑
        batch_model = np.linspace(batch_grid[0], batch_grid[-1], model_resolution[1])
        token_model = np.linspace(token_grid[0], token_grid[-1], model_resolution[0])
        batch_mesh_model, token_mesh_model = np.meshgrid(batch_model, token_model)
        
        # 2. 使用拟合模型生成理论预测值
        runtime_theory = self.runtime_model_function(
            batch_mesh_model, token_mesh_model, fitted_params
        )
        
        # 3. 设置理论等高线级别
        theory_min = np.min(runtime_theory)
        theory_max = np.max(runtime_theory)
        # 使用合理的等高线间隔
        contour_levels = np.linspace(theory_min, theory_max, 10)
        
        # 4. 绘制基于理论模型的平滑等高线
        contours = ax.contour(
            batch_mesh_model, token_mesh_model, runtime_theory,
            levels=contour_levels,
            colors='white',
            linewidths=1.2,
            alpha=0.8,
            linestyles='-'
        )
        
        # 5. 添加等高线标签
        ax.clabel(contours, inline=True, fontsize=14, fmt='%.0f', colors='white')
        
        # 设置坐标轴标签，参考draw_chunk_runtime的样式
        ax.set_xlabel('Batch Size', fontsize=23, labelpad=10)  # 参考draw_chunk_runtime
        ax.set_ylabel('Total Scheduled Tokens', fontsize=23, labelpad=10)  # 参考draw_chunk_runtime
        
        # 设置坐标轴刻度 - 确保显示具体数值而非范围
        # Batch Size刻度
        batch_ticks = np.linspace(batch_grid[0], batch_grid[-1], 6).astype(int)
        ax.set_xticks(batch_ticks)
        ax.set_xticklabels(batch_ticks)
        
        # Total Tokens刻度 - 使用固定刻度值
        token_ticks = [1, 1024, 2048, 3072, 4096]
        ax.set_yticks(token_ticks)
        ax.set_yticklabels(token_ticks)
        
        # 设置刻度字体大小，参考draw_chunk_runtime
        ax.tick_params(axis='both', pad=8, labelsize=19)  # 参考draw_chunk_runtime
        
        # 设置网格，参考draw_chunk_runtime
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=2)  # 参考draw_chunk_runtime
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Model Run Time (ms)', fontsize=16, labelpad=5)
        cbar.ax.tick_params(labelsize=14)
        
        # 调整布局
        plt.subplots_adjust(left=0.12, right=0.85, bottom=0.15, top=0.92)
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            if self.verbose:
                print(f"✅ 热力图已保存: {save_path}")
        
        return fig
    
    def run_end_to_end(self, log_path: str, save_path: str = None, 
                      resolution: tuple = (25, 25)) -> bool:
        """端到端运行热力图生成"""
        try:
            if self.verbose:
                print("🚀 开始生成Batch Size vs Total Tokens热力图")
                print("=" * 60)
            
            # 1. 读取数据
            df = self.read_profiling_data(log_path)
            if df is None or len(df) == 0:
                if self.verbose:
                    print("❌ 无法读取有效数据")
                return False
            
            # 2. 提取特征
            valid_df = self.extract_batch_and_token_features(df)
            if len(valid_df) < 10:
                if self.verbose:
                    print("⚠️ 有效数据点太少，可能影响图表质量")
            
            # 3. 创建热力图数据
            runtime_matrix, batch_grid, token_grid, fitted_params, r_squared = self.create_heatmap_data(
                valid_df, resolution=resolution
            )
            
            # 4. 绘制热力图
            fig = self.plot_heatmap(
                runtime_matrix, batch_grid, token_grid, fitted_params, r_squared, save_path
            )
            
            if self.verbose:
                print("\n🎉 热力图生成完成！")
                print("=" * 60)
                print("📐 模型拟合结果:")
                print(f"   R² = {r_squared:.4f}")
                print(f"   线性系数 (α): {fitted_params['alpha']:.6f}")
                print(f"   饱和最大值 (γ): {fitted_params['gamma']:.2f}")
                print(f"   饱和参数 (B0): {fitted_params['B0']:.2f}")
                print(f"📊 模型公式: T(B,S) = {fitted_params['base_time']:.2f} + {fitted_params['alpha']:.6f}×S + {fitted_params['gamma']:.2f}×(1-e^(-B/{fitted_params['B0']:.2f}))")
            
            plt.show()
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 生成热力图时出错: {e}")
                traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成Batch Size vs Total Tokens热力图')
    parser.add_argument('log_path', type=str, nargs='?', 
                       default='scheduler_profiling.jsonl',
                       help='profiling数据文件或目录路径')
    parser.add_argument('--save-path', type=str, 
                       default="./paper_figs/batch_token_heatmap_fig/batch_token_heatmap.pdf",
                       help='图表保存路径')
    parser.add_argument('--resolution', type=int, nargs=2, default=[10, 10],
                       help='热力图分辨率 (batch_bins token_bins)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    generator = BatchTokenHeatmapGenerator(verbose=args.verbose)
    
    success = generator.run_end_to_end(
        log_path=args.log_path,
        save_path=args.save_path,
        resolution=tuple(args.resolution)
    )
    
    if success:
        print("✅ 热力图生成完成！")
    else:
        print("❌ 热力图生成失败！")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 