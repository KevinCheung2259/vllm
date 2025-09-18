#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于生成 "Predicted vs Actual (Stable Model)" 图表
实现从数据读取、模型训练到图表生成的端到端功能
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# 添加modeling目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'modeling'))

try:
    from performance_model import StableClusterModel
except ImportError:
    # 如果导入失败，尝试从绝对路径导入
    sys.path.insert(0, '/home/paperspace/zhangy/vllm-workspace/vllm/modeling')
    try:
        from performance_model import StableClusterModel
    except ImportError:
        print("❌ 无法导入StableClusterModel，请确保路径正确")
        sys.exit(1)


class PredictedVsActualGenerator:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.models = []  # 存储多个模型
        self.dfs = []     # 存储多个数据集
        self.labels = []  # 存储每个数据集的标签
        self.setup_style()

    def setup_style(self):
        """设置matplotlib样式，优化PDF渲染性能并与其他图风格一致"""
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            # 降低矢量路径复杂度，提升PDF渲染速度
            'path.simplify': True,
            'path.simplify_threshold': 0.5,
            'agg.path.chunksize': 10000,
            'pdf.compression': 9,
        })
    
    def read_profiling_data(self, log_file_or_dir: str) -> pd.DataFrame:

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
        data = [item for item in data if item.get('schedule_duration_ms', 0) <200]
        data = [item for item in data if item.get('model_run_duration_ms', 0) < 200]
        
        df = pd.DataFrame(data)
        
        # 计算decode和prefill请求数
        if 'chunk_sizes' in df.columns:
            def _count_decode_reqs(sizes):
                if isinstance(sizes, list):
                    return sum(1 for s in sizes if s == 1)
                try:
                    return 1 if sizes == 1 else 0
                except Exception:
                    return 0
            
            def _count_prefill_reqs(sizes):
                if isinstance(sizes, list):
                    return sum(1 for s in sizes if s > 1)
                try:
                    return 1 if sizes > 1 else 0
                except Exception:
                    return 0
            
            df['num_decode_reqs'] = df['chunk_sizes'].apply(_count_decode_reqs)
            df['num_prefill_reqs'] = df['chunk_sizes'].apply(_count_prefill_reqs)
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> StableClusterModel:
        """训练StableClusterModel"""

        if self.verbose:
            print("🚀 开始训练稳定集群调度模型...")
        
        model = StableClusterModel(verbose=self.verbose)
        model.fit(df)
        
        if self.verbose:
            print("✅ 模型训练完成")
        
        return model
    
    def generate_multi_dataset_plot(self, models, dfs, labels=None,
                                   save_path: str = None, ranges:list[list] = None,
                                   rasterized: bool = False,
                                   max_points_per_dataset: Optional[int] = None,
                                   point_size: int = 50,
                                   alpha: float = 0.6,
                                   remove_edgecolors: bool = False) -> plt.Figure:
        """
        生成包含多个数据集的"Predicted vs Actual (Stable Model)"图表
        
        Args:
            models: 训练好的StableClusterModel实例列表
            dfs: 包含profiling数据的DataFrame列表
            labels: 数据集标签列表（可选）
            colors: 数据点颜色列表（可选）
            save_path: 图表保存路径（可选）
            
        Returns:
            matplotlib Figure对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        # 用于确定坐标轴范围
        all_T = []
        all_T_pred = []
        
        # 默认颜色循环
        default_colors = plt.cm.tab10.colors
        # 默认标记样式循环
        default_markers = ['o', 's', '^', 'v', 'D', '*', '+', 'x', 'p', 'h']
        
        # 为每个数据集绘制散点图
        for i, (model, df) in enumerate(zip(models, dfs)):
            # 获取数据
            B, S, T = model._preprocess_data(df)

            B_norm, S_norm, _ = model._normalize_features(B, S)
            
            # 预测值
            T_pred = model.stable_latency_model((B_norm, S_norm), *model.params, model.P_max)
            
            # 添加过滤逻辑：删除T中数值大于100的数据点，同时删除对应的T_pred
            # 将T和T_pred转换为numpy数组以便进行向量化操作
            import numpy as np
            T = np.array(T)
            T_pred = np.array(T_pred)
            
            # 应用 mask_1 过滤数据
            mask_1 = (T>ranges[i][0]) & (T<ranges[i][1])
            T_1 = T[mask_1]
            T_pred_1 = T_pred[mask_1]

            mask_2 = (T_pred_1>ranges[i][0]) & (T_pred_1<ranges[i][1])
            T_2 = T_1[mask_2]
            T_pred_2 = T_pred_1[mask_2]
            
            # 下采样：按数据集限制最大点数
            if max_points_per_dataset is not None and len(T_2) > max_points_per_dataset:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(T_2), size=max_points_per_dataset, replace=False)
                T_2 = T_2[idx]
                T_pred_2 = T_pred_2[idx]

            # 如果该数据集在过滤/下采样后为空，则跳过
            if len(T_2) == 0:
                continue

            # 存储过滤后的值以确定范围
            all_T.extend(T_2)
            all_T_pred.extend(T_pred_2)


            # # 存储所有值以确定范围
            # all_T.extend(T)
            # all_T_pred.extend(T_pred)
            
            # 获取标签、颜色和标记
            label = labels[i]
            color = default_colors[i % len(default_colors)]
            marker = default_markers[i % len(default_markers)]
            
            # 绘制散点图（可选：栅格化、去边框）
            edgecolor_value = 'none' if remove_edgecolors else 'white'
            linewidth_value = 0.0 if remove_edgecolors else 0.5
            ax.scatter(
                T_pred_2,
                T_2,
                alpha=alpha,
                s=point_size,
                c=[color],
                marker=marker,
                label=label,
                edgecolors=edgecolor_value,
                linewidth=linewidth_value,
                rasterized=rasterized,
            )
        
        # 理想线 (y=x)
        min_val = min(min(all_T), min(all_T_pred))
        max_val = max(max(all_T), max(all_T_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=4, label='Ideal (y=x)')
        
        # 设置图表属性
        ax.set_xlabel('Predicted Latency (ms)', fontsize=23,labelpad=10)
        ax.set_ylabel('Actual Latency (ms)', fontsize=23,labelpad=10)
        # ax.set_title('Predicted vs Actual latency', fontsize=18,pad=20)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=2)
        
        # 添加图例
        ax.legend(loc='upper left',fontsize = 18,markerscale=1.8)
        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', pad=8, labelsize=19)  
        # 控制绘图区域与图片上下左右的间距
        plt.subplots_adjust(left=0.15, right=0.8, bottom=0.15, top=0.8)

        plt.tight_layout()
        
        # 保存图表（如果提供了保存路径）
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"📊 图表已保存至: {save_path}")
        
            pdf_path = save_path.replace(".png",".pdf")
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')

    
        return fig
    
    def run_multi_dataset_end_to_end(self, log_paths: list, save_path: str = None, 
                                     labels: list = None, ranges:list[list] = None,
                                     rasterized: bool = False,
                                     max_points_per_dataset: Optional[int] = None,
                                     point_size: int = 50,
                                     alpha: float = 0.6,
                                     remove_edgecolors: bool = False) -> bool:
        """
        运行端到端的流程处理多个数据集：读取数据、训练模型、生成合并图表
        
        Args:
            log_paths: 日志文件或目录路径列表
            save_path: 图表保存路径（可选）
            labels: 数据集标签列表（可选）
            
        Returns:
            是否成功执行
        """
        try:
            models = []
            dfs = []
            
            # 为每个数据集读取数据并训练模型
            for i, log_path in enumerate(log_paths):
                # 1.读取数据
                if self.verbose:
                    print(f"\n📥 正在读取数据集 {i+1}/{len(log_paths)} 的profiling数据...")

                # 获取当前数据集的清洗阈值，如果没有提供则使用默认值
                
                df = self.read_profiling_data(log_path)
                if df is None or df.empty:
                    print(f"❌ 数据集 {i+1} 未找到有效数据，跳过")
                    continue
                dfs.append(df)
                
                if self.verbose:
                    print(f"✅ 成功读取 {len(df)} 条profiling数据")
                
                # 2.训练模型
                model = self.train_model(df)
                models.append(model)
            
            if not models or not dfs:
                print("❌ 没有成功加载任何数据集")
                return False
            
            # 保存模型和数据以便后续使用
            self.models = models
            self.dfs = dfs
            if labels:
                self.labels = labels
            
            # 3.生成多数据集图表
            if self.verbose:
                print("\n📊 正在生成包含多个数据集的'Predicted vs Actual'图表...")
            self.generate_multi_dataset_plot(
                models,
                dfs,
                labels,
                save_path=save_path,
                ranges=ranges,
                rasterized=rasterized,
                max_points_per_dataset=max_points_per_dataset,
                point_size=point_size,
                alpha=alpha,
                remove_edgecolors=remove_edgecolors,
            )
            
            return True
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():

    parser = argparse.ArgumentParser(description='生成Predicted vs Actual (Stable Model)图表')
    parser.add_argument('log_path', type=str, nargs='*',
                      help='profiling数据文件或目录路径 (可指定多个，默认: profiling_result)')
    parser.add_argument('--save-path', type=str, default="./predicted_and_actual_latency.pdf")
    parser.add_argument('--labels', type=str, nargs='*',help='为每个数据集指定标签 (与log_path顺序对应)')
    parser.add_argument('--rasterized', action='store_true', help='将散点以栅格方式嵌入PDF，显著降低PDF渲染开销')
    parser.add_argument('--max-points-per-dataset', type=int, default=None, help='每个数据集最多绘制的点数，超出将随机下采样')
    parser.add_argument('--point-size', type=int, default=50, help='散点大小')
    parser.add_argument('--alpha', type=float, default=0.6, help='散点透明度')
    parser.add_argument('--no-edges', action='store_true', help='移除散点边框以减少矢量路径复杂度')
    
    args = parser.parse_args()
    
    generator = PredictedVsActualGenerator()
    base_dir = "/home/paperspace/zhangy/vllm-workspace/vllm/exp"
    default_data = {
        "H100":{"log_path":f"{base_dir}/profiling_result_h100","T_range":[0,200]},
        "A100":{"log_path":f"{base_dir}/profiling_result_a100","T_range":[100,200]},
        "A6000-TP2":{"log_path":f"{base_dir}/profiling_result_a6000","T_range":[100,200]},
        "H100-32B":{"log_path":f"{base_dir}/profiling_result_h100_qwen32b","T_range":[25,200]},
    }

    labels = list(default_data.keys())  # 获取所有标签
    log_paths = [default_data[label]["log_path"] for label in default_data]
    ranges = [default_data[label]["T_range"] for label in default_data]

    # 多个数据集模式
    success = generator.run_multi_dataset_end_to_end(
        log_paths=args.log_path if args.log_path else log_paths,
        save_path=args.save_path,
        labels=args.labels if args.labels else labels,
        ranges = ranges,
        rasterized=args.rasterized,
        max_points_per_dataset=args.max_points_per_dataset,
        point_size=args.point_size,
        alpha=args.alpha,
        remove_edgecolors=args.no_edges,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()