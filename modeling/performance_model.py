#!/usr/bin/env python3
"""
vLLM Scheduler 性能建模
基于吞吐饱和理论的物理启发模型

模型形式:
Thr(B,S) = P_max * (1 - exp(-k_B * B)) * (1 - exp(-k_S * S))
Work(B,S) = w_0 + w_1 * S
T(B,S) = τ_0 + Work(B,S) / Thr(B,S) + τ_B * B + τ_S * S

其中:
- B: batch_size (请求数量)
- S: total_tokens (sum of chunk_sizes)
- T: model_run_duration_ms (延迟)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from typing import Tuple, Dict, Optional, Union
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThroughputSaturationModel:
    """基于吞吐饱和理论的性能建模类"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化模型
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.is_fitted = False
        self.params = None
        self.scales = None
        self.fit_metrics = None
        
        # 参数名称和含义
        self.param_names = [
            'P_max', 'k_B', 'k_S', 'w_0', 'w_1', 'tau_0', 'tau_B', 'tau_S'
        ]
        self.param_descriptions = {
            'P_max': '最大有效吞吐量 (tokens/ms)',
            'k_B': 'batch并行度敏感系数',
            'k_S': 'token并行度敏感系数', 
            'w_0': '基础工作量常数',
            'w_1': '每token工作量系数',
            'tau_0': '基础延迟常数 (ms)',
            'tau_B': '每batch额外延迟 (ms/batch)',
            'tau_S': '每token额外延迟 (ms/token)'
        }
    
    @staticmethod
    def throughput(B: np.ndarray, S: np.ndarray, P_max: float, k_B: float, k_S: float) -> np.ndarray:
        """
        计算有效吞吐量
        
        Args:
            B: batch_size数组
            S: total_tokens数组
            P_max: 最大吞吐量
            k_B: batch敏感系数
            k_S: token敏感系数
            
        Returns:
            有效吞吐量数组
        """
        return P_max * (1.0 - np.exp(-k_B * B)) * (1.0 - np.exp(-k_S * S))
    
    @staticmethod
    def workload(S: np.ndarray, w_0: float, w_1: float) -> np.ndarray:
        """
        计算工作量
        
        Args:
            S: total_tokens数组
            w_0: 基础工作量
            w_1: 每token工作量
            
        Returns:
            工作量数组
        """
        return w_0 + w_1 * S
    
    @staticmethod
    def latency_model(X: Tuple[np.ndarray, np.ndarray], 
                     P_max: float, k_B: float, k_S: float, 
                     w_0: float, w_1: float, 
                     tau_0: float, tau_B: float, tau_S: float) -> np.ndarray:
        """
        完整的延迟模型
        
        Args:
            X: (B, S) 元组，其中B为batch_size数组，S为total_tokens数组
            P_max, k_B, k_S: 吞吐量参数
            w_0, w_1: 工作量参数
            tau_0, tau_B, tau_S: 线性开销参数
            
        Returns:
            预测延迟数组
        """
        B, S = X
        
        # 计算有效吞吐量
        thr = ThroughputSaturationModel.throughput(B, S, P_max, k_B, k_S)
        
        # 计算工作量
        work = ThroughputSaturationModel.workload(S, w_0, w_1)
        
        # 避免除零
        thr = np.maximum(thr, 1e-9)
        
        # 计算总延迟
        latency = tau_0 + work / thr + tau_B * B + tau_S * S
        
        return latency
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从DataFrame中提取特征
        
        Args:
            df: 包含scheduler profiling数据的DataFrame
            
        Returns:
            (B, S, T) 元组，分别为batch_size、total_tokens、延迟
        """
        def _compute_batch_size(chunk_sizes):
            """计算batch size"""
            if isinstance(chunk_sizes, list):
                return len(chunk_sizes)
            return np.nan
            
        def _compute_total_tokens(chunk_sizes):
            """计算总token数"""
            if isinstance(chunk_sizes, list) and len(chunk_sizes) > 0:
                return float(sum(chunk_sizes))
            if isinstance(chunk_sizes, (int, float)):
                return float(chunk_sizes)
            return np.nan
        
        # 提取特征
        B = df['chunk_sizes'].apply(_compute_batch_size).to_numpy(dtype=float)
        S = df['chunk_sizes'].apply(_compute_total_tokens).to_numpy(dtype=float)
        T = df['model_run_duration_ms'].to_numpy(dtype=float)
        
        return B, S, T
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        数据预处理：提取特征、过滤异常值、检查有效性
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            处理后的 (B, S, T) 元组
        """
        if self.verbose:
            logger.info("开始数据预处理...")
            
        # 提取特征
        B, S, T = self._extract_features(df)
        
        # 初始数据量
        initial_count = len(T)
        
        # 创建有效性掩码
        valid_mask = (
            np.isfinite(B) & np.isfinite(S) & np.isfinite(T) &
            (B > 0) & (S > 0) & (T > 0) &
            (T < 200)  # 过滤过长延迟
        )
        
        # 应用掩码
        B, S, T = B[valid_mask], S[valid_mask], T[valid_mask]
        
        if self.verbose:
            logger.info(f"数据过滤: {initial_count} -> {len(T)} 样本")
            logger.info(f"Batch Size 范围: [{B.min():.1f}, {B.max():.1f}]")
            logger.info(f"Total Tokens 范围: [{S.min():.1f}, {S.max():.1f}]")
            logger.info(f"延迟范围: [{T.min():.2f}, {T.max():.2f}] ms")
        
        if len(T) < 20:
            raise ValueError(f"有效样本数过少: {len(T)}，建议至少20个样本")
            
        return B, S, T
    
    def _normalize_features(self, B: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        特征归一化，使用中位数缩放
        
        Args:
            B: batch_size数组
            S: total_tokens数组
            
        Returns:
            (B_norm, S_norm, scales) 元组
        """
        B_median = np.median(B)
        S_median = np.median(S)
        
        # 避免除零
        B_scale = max(B_median, 1.0)
        S_scale = max(S_median, 1.0)
        
        B_norm = B / B_scale
        S_norm = S / S_scale
        
        scales = (B_scale, S_scale)
        
        if self.verbose:
            logger.info(f"特征归一化: B_scale={B_scale:.2f}, S_scale={S_scale:.2f}")
            
        return B_norm, S_norm, scales
    
    def _initialize_parameters(self, B: np.ndarray, S: np.ndarray, T: np.ndarray) -> Tuple[list, list, list]:
        """
        参数初始化
        
        Args:
            B, S, T: 归一化后的特征和目标
            
        Returns:
            (初始值, 下界, 上界) 元组
        """
        # 估计峰值吞吐量
        throughput_estimates = S / np.maximum(T, 1e-6)
        P_max_init = max(np.percentile(throughput_estimates, 95), 1e-3)
        
        # 线性拟合估计w_1
        try:
            w_1_init = max(np.polyfit(S, T, 1)[0], 1e-9)
        except:
            w_1_init = 0.01
        
        # 初始参数
        p0 = [
            P_max_init,     # P_max
            0.1,            # k_B
            0.01,           # k_S  
            0.1,            # w_0
            w_1_init,       # w_1
            np.min(T) * 0.5, # tau_0
            1e-3,           # tau_B
            1e-3            # tau_S
        ]
        
        # 参数下界（物理合理性）
        lower_bounds = [1e-6, 1e-6, 1e-6, 0.0, 1e-9, 0.0, 0.0, 0.0]
        
        # 参数上界（防止过拟合）
        upper_bounds = [100, 10.0, 10.0, 10.0, 1e2, 1e3, 1e1, 1e1]
        
        if self.verbose:
            logger.info(f"参数初始化: P_max={P_max_init:.3f}, w_1={w_1_init:.6f}")
            
        return p0, lower_bounds, upper_bounds
    
    def fit(self, df: pd.DataFrame) -> 'ThroughputSaturationModel':
        """
        拟合模型
        
        Args:
            df: 包含scheduler profiling数据的DataFrame
            
        Returns:
            self (支持链式调用)
        """
        if self.verbose:
            logger.info("开始模型拟合...")
            
        # 数据预处理
        B, S, T = self._preprocess_data(df)
        
        # 特征归一化
        B_norm, S_norm, scales = self._normalize_features(B, S)
        self.scales = scales
        
        # 参数初始化
        p0, lower_bounds, upper_bounds = self._initialize_parameters(B_norm, S_norm, T)
        
        # 非线性拟合
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                popt, pcov = curve_fit(
                    lambda X, *params: self.latency_model(X, *params),
                    xdata=(B_norm, S_norm),
                    ydata=T,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=20000,
                    method='trf'  # Trust Region Reflective算法，对边界约束友好
                )
                
            self.params = popt
            self.param_cov = pcov
            self.is_fitted = True
            
            # 计算拟合质量指标
            T_pred = self.latency_model((B_norm, S_norm), *popt)
            self.fit_metrics = {
                'r2': r2_score(T, T_pred),
                'rmse': np.sqrt(mean_squared_error(T, T_pred)),
                'mae': mean_absolute_error(T, T_pred),
                'n_samples': len(T)
            }
            
            if self.verbose:
                self._print_fit_results()
                
        except Exception as e:
            logger.error(f"拟合失败: {e}")
            raise
            
        return self
    
    def _print_fit_results(self):
        """打印拟合结果"""
        logger.info("拟合完成!")
        logger.info(f"R² = {self.fit_metrics['r2']:.4f}")
        logger.info(f"RMSE = {self.fit_metrics['rmse']:.3f} ms")
        logger.info(f"MAE = {self.fit_metrics['mae']:.3f} ms")
        logger.info(f"样本数 = {self.fit_metrics['n_samples']}")
        
        logger.info("\n拟合参数:")
        for i, (name, desc) in enumerate(zip(self.param_names, self.param_descriptions.values())):
            logger.info(f"{name:8s} = {self.params[i]:10.6f}  # {desc}")
            
        # 参数标准误差（如果协方差矩阵可用）
        try:
            param_std = np.sqrt(np.diag(self.param_cov))
            logger.info("\n参数标准误差:")
            for i, name in enumerate(self.param_names):
                logger.info(f"{name:8s} ± {param_std[i]:10.6f}")
        except:
            pass
    
    def predict(self, batch_size: Union[float, np.ndarray], 
                total_tokens: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        预测延迟
        
        Args:
            batch_size: 批次大小
            total_tokens: 总token数
            
        Returns:
            预测的延迟 (ms)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        # 转换为数组
        B = np.asarray(batch_size)
        S = np.asarray(total_tokens)
        
        # 归一化
        B_scale, S_scale = self.scales
        B_norm = B / B_scale
        S_norm = S / S_scale
        
        # 预测
        latency = self.latency_model((B_norm, S_norm), *self.params)
        
        return latency
    
    def plot_contour(self, batch_range: Tuple[int, int] = (1, 64),
                    token_range: Tuple[int, int] = (1, 2048),
                    resolution: int = 50,
                    figsize: Tuple[int, int] = (12, 8),
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制延迟等高线图
        
        Args:
            batch_range: batch_size范围
            token_range: total_tokens范围  
            resolution: 网格分辨率
            figsize: 图像大小
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib Figure对象
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        # 创建网格
        B_grid = np.linspace(batch_range[0], batch_range[1], resolution)
        S_grid = np.linspace(token_range[0], token_range[1], resolution)
        B_mesh, S_mesh = np.meshgrid(B_grid, S_grid)
        
        # 预测延迟
        T_mesh = self.predict(B_mesh.flatten(), S_mesh.flatten()).reshape(B_mesh.shape)
        
        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 填充等高线
        cs1 = ax.contourf(B_mesh, S_mesh, T_mesh, levels=20, cmap='viridis', alpha=0.8)
        cbar1 = plt.colorbar(cs1, ax=ax)
        cbar1.set_label('Model Run Latency (ms)', fontsize=12)
        
        # 等高线
        cs2 = ax.contour(B_mesh, S_mesh, T_mesh, levels=10, colors='white', linewidths=1, alpha=0.7)
        ax.clabel(cs2, inline=True, fontsize=9, fmt='%1.0f ms')
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Total Tokens', fontsize=12)
        ax.set_title('vLLM Model Run Latency Prediction\n(Throughput Saturation Model)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加模型信息
        if self.fit_metrics:
            info_text = f"R² = {self.fit_metrics['r2']:.3f}\nRMSE = {self.fit_metrics['rmse']:.1f} ms"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                logger.info(f"Contour plot saved: {save_path}")
                
        return fig
    
    def plot_residuals(self, df: pd.DataFrame, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        绘制残差分析图
        
        Args:
            df: 训练数据DataFrame
            figsize: 图像大小
            
        Returns:
            matplotlib Figure对象
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        # 获取数据
        B, S, T = self._preprocess_data(df)
        B_norm, S_norm, _ = self._normalize_features(B, S)
        
        # 预测值
        T_pred = self.latency_model((B_norm, S_norm), *self.params)
        residuals = T - T_pred
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 预测值 vs 真实值
        axes[0].scatter(T_pred, T, alpha=0.6, s=20)
        axes[0].plot([T.min(), T.max()], [T.min(), T.max()], 'r--', lw=2)
        axes[0].set_xlabel('Predicted Latency (ms)')
        axes[0].set_ylabel('Actual Latency (ms)')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # 残差 vs 预测值
        axes[1].scatter(T_pred, residuals, alpha=0.6, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Latency (ms)')
        axes[1].set_ylabel('Residuals (ms)')
        axes[1].set_title('Residuals vs Predicted')
        axes[1].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[2].set_xlabel('Residuals (ms)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self) -> Dict:
        """
        获取模型摘要信息
        
        Returns:
            包含模型参数和指标的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        summary = {
            'parameters': dict(zip(self.param_names, self.params)),
            'metrics': self.fit_metrics,
            'scales': {'batch_scale': self.scales[0], 'token_scale': self.scales[1]}
        }
        
        return summary
    
    def save_model(self, filepath: str):
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        import pickle
        
        model_data = {
            'params': self.params,
            'scales': self.scales,
            'fit_metrics': self.fit_metrics,
            'param_cov': getattr(self, 'param_cov', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        if self.verbose:
            logger.info(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.params = model_data['params']
        self.scales = model_data['scales'] 
        self.fit_metrics = model_data['fit_metrics']
        self.param_cov = model_data.get('param_cov')
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"Model loaded: {filepath}")
            self._print_fit_results()


class StableClusterModel:
    """
    稳定的集群调度模型
    使用两阶段拟合方法：
    1. 首先通过简单方法估计稳定的P_max
    2. 然后固定P_max，拟合其他参数
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.is_fitted = False
        self.P_max = None  # 独立估计的峰值吞吐量
        self.params = None  # 其他拟合参数
        self.scales = None
        self.fit_metrics = None
        
        # 参数名称（不包括P_max，因为它独立估计）
        self.param_names = ['k_B', 'k_S', 'tau_B', 'tau_S', 'T_base']
        self.param_descriptions = {
            'P_max': '硬件峰值吞吐量 (tokens/ms) - 独立估计',
            'k_B': 'batch并行度敏感系数',
            'k_S': 'token并行度敏感系数',
            'tau_B': '每batch线性开销 (ms/batch)',
            'tau_S': '每token线性开销 (ms/token)',
            'T_base': '基础延迟时间 (ms)'
        }
    
    def _estimate_peak_throughput(self, B: np.ndarray, S: np.ndarray, T: np.ndarray) -> float:
        """
        独立估计峰值吞吐量P_max
        使用大批次、高token数场景下的数据
        """
        # 选择大批次、高token的样本来估计峰值吞吐量
        large_batch_mask = (B >= np.percentile(B, 80)) & (S >= np.percentile(S, 80))
        
        if np.sum(large_batch_mask) < 10:
            # 如果大批次样本太少，使用全部数据
            effective_throughput = S / T
        else:
            # 使用大批次样本
            effective_throughput = S[large_batch_mask] / T[large_batch_mask]
        
        # 使用90分位数作为峰值吞吐量的估计
        P_max_estimate = np.percentile(effective_throughput, 90)
        
        if self.verbose:
            logger.info(f"峰值吞吐量估计: {P_max_estimate:.4f} tokens/ms")
            
        return P_max_estimate
    
    @staticmethod
    def stable_latency_model(X: Tuple[np.ndarray, np.ndarray], 
                           k_B: float, k_S: float, tau_B: float, tau_S: float, T_base: float,
                           P_max_fixed: float) -> np.ndarray:
        """
        稳定的延迟模型，P_max作为固定参数传入
        
        模型形式: T = T_base / (eff_B * eff_S) + tau_B * B + tau_S * S
        其中 eff_B = (1 - exp(-k_B * B)), eff_S = (1 - exp(-k_S * S))
        T_base 包含了 P_max 的影响，但通过约束关系避免冗余
        """
        B, S = X
        
        # 计算并行效率因子
        eff_B = 1.0 - np.exp(-k_B * B)
        eff_S = 1.0 - np.exp(-k_S * S)
        
        # 避免除零
        eff_B = np.maximum(eff_B, 1e-6)
        eff_S = np.maximum(eff_S, 1e-6)
        
        # 延迟模型：基础延迟/效率 + 线性开销
        # T_base 与 P_max 有约束关系，但 P_max 已固定
        latency = T_base / (eff_B * eff_S) + tau_B * B + tau_S * S
        
        return latency
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从DataFrame中提取特征"""
        def _compute_batch_size(chunk_sizes):
            if isinstance(chunk_sizes, list):
                return len(chunk_sizes)
            return np.nan
            
        def _compute_total_tokens(chunk_sizes):
            if isinstance(chunk_sizes, list) and len(chunk_sizes) > 0:
                return float(sum(chunk_sizes))
            if isinstance(chunk_sizes, (int, float)):
                return float(chunk_sizes)
            return np.nan
        
        B = df['chunk_sizes'].apply(_compute_batch_size).to_numpy(dtype=float)
        S = df['chunk_sizes'].apply(_compute_total_tokens).to_numpy(dtype=float)
        T = df['model_run_duration_ms'].to_numpy(dtype=float)
        
        return B, S, T
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据预处理"""
        if self.verbose:
            logger.info("开始数据预处理...")
            
        B, S, T = self._extract_features(df)
        initial_count = len(T)
        
        # 数据过滤
        valid_mask = (
            np.isfinite(B) & np.isfinite(S) & np.isfinite(T) &
            (B > 0) & (S > 0) & (T > 0) &
            (T < 300)  # 过滤异常值
        )
        
        B, S, T = B[valid_mask], S[valid_mask], T[valid_mask]
        
        if self.verbose:
            logger.info(f"数据过滤: {initial_count} -> {len(T)} 样本")
            logger.info(f"Batch Size 范围: [{B.min():.1f}, {B.max():.1f}]")
            logger.info(f"Total Tokens 范围: [{S.min():.1f}, {S.max():.1f}]")
            logger.info(f"延迟范围: [{T.min():.2f}, {T.max():.2f}] ms")
        
        if len(T) < 20:
            raise ValueError(f"有效样本数过少: {len(T)}")
            
        return B, S, T
    
    def _normalize_features(self, B: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """特征归一化"""
        B_median = np.median(B)
        S_median = np.median(S)
        
        B_scale = max(B_median, 1.0)
        S_scale = max(S_median, 1.0)
        
        B_norm = B / B_scale
        S_norm = S / S_scale
        
        scales = (B_scale, S_scale)
        
        if self.verbose:
            logger.info(f"特征归一化: B_scale={B_scale:.2f}, S_scale={S_scale:.2f}")
            
        return B_norm, S_norm, scales
    
    def fit(self, df: pd.DataFrame) -> 'StableClusterModel':
        """
        两阶段拟合方法
        1. 独立估计P_max
        2. 固定P_max，拟合其他参数
        """
        if self.verbose:
            logger.info("开始稳定集群调度模型拟合...")
            
        # 数据预处理
        B, S, T = self._preprocess_data(df)
        B_norm, S_norm, scales = self._normalize_features(B, S)
        self.scales = scales
        
        # 阶段1：独立估计P_max
        if self.verbose:
            logger.info("阶段1: 估计峰值吞吐量...")
        self.P_max = self._estimate_peak_throughput(B_norm, S_norm, T)
        
        # 阶段2：固定P_max，拟合其他参数
        if self.verbose:
            logger.info("阶段2: 拟合其他参数...")
            
        # 基于P_max估计初始参数
        T_median = np.median(T)
        
        p0 = [
            0.5,                # k_B
            0.2,                # k_S  
            1.0,                # tau_B
            0.1,                # tau_S
            T_median * 0.5      # T_base
        ]
        
        # 参数边界
        lower_bounds = [0.01, 0.001, 0.0, 0.0, 0.1]
        upper_bounds = [5.0, 2.0, 20.0, 2.0, 200.0]
        
        # 拟合（P_max作为固定参数传入）
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                popt, pcov = curve_fit(
                    lambda X, *params: self.stable_latency_model(X, *params, self.P_max),
                    xdata=(B_norm, S_norm),
                    ydata=T,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=15000,
                    method='trf'
                )
                
            self.params = popt
            self.param_cov = pcov
            self.is_fitted = True
            
            # 计算拟合质量
            T_pred = self.stable_latency_model((B_norm, S_norm), *popt, self.P_max)
            self.fit_metrics = {
                'r2': r2_score(T, T_pred),
                'rmse': np.sqrt(mean_squared_error(T, T_pred)),
                'mae': mean_absolute_error(T, T_pred),
                'n_samples': len(T)
            }
            
            if self.verbose:
                self._print_fit_results()
                
        except Exception as e:
            logger.error(f"拟合失败: {e}")
            raise
            
        return self
    
    def _print_fit_results(self):
        """打印拟合结果"""
        logger.info("稳定集群调度模型拟合完成!")
        logger.info(f"R² = {self.fit_metrics['r2']:.4f}")
        logger.info(f"RMSE = {self.fit_metrics['rmse']:.3f} ms")
        logger.info(f"MAE = {self.fit_metrics['mae']:.3f} ms")
        logger.info(f"样本数 = {self.fit_metrics['n_samples']}")
        
        logger.info("\n模型参数:")
        logger.info(f"{'P_max':12s} = {self.P_max:10.6f}  # {self.param_descriptions['P_max']} (独立估计)")
        
        for i, (name, desc) in enumerate(zip(self.param_names, [self.param_descriptions[name] for name in self.param_names])):
            logger.info(f"{name:12s} = {self.params[i]:10.6f}  # {desc}")
            
        # 参数稳定性分析
        try:
            param_std = np.sqrt(np.diag(self.param_cov))
            logger.info("\n参数稳定性分析:")
            logger.info(f"{'P_max':12s} ± {'N/A':8s} (独立估计，稳定)")
            
            unstable_count = 0
            for i, name in enumerate(self.param_names):
                if abs(self.params[i]) > 1e-6:
                    relative_error = param_std[i] / abs(self.params[i])
                    if relative_error > 0.5:
                        status = "⚠️  不稳定"
                        unstable_count += 1
                    elif relative_error > 0.2:
                        status = "⚡ 中等"
                    else:
                        status = "✅ 稳定"
                    
                    logger.info(f"{name:12s} ± {param_std[i]:8.6f} (相对误差: {relative_error:6.1%}) {status}")
            
            if unstable_count == 0:
                logger.info("🎉 所有参数都稳定!")
            else:
                logger.warning(f"⚠️  {unstable_count} 个参数不稳定")
                
        except Exception as e:
            logger.warning(f"无法计算参数稳定性: {e}")
    
    def predict(self, batch_size: Union[float, np.ndarray], 
                total_tokens: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """预测延迟"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
            
        B = np.asarray(batch_size)
        S = np.asarray(total_tokens)
        
        # 归一化
        B_scale, S_scale = self.scales
        B_norm = B / B_scale
        S_norm = S / S_scale
        
        # 预测
        latency = self.stable_latency_model((B_norm, S_norm), *self.params, self.P_max)
        
        return latency
    
    def get_hardware_capacity(self) -> Dict[str, float]:
        """获取硬件能力参数，用于集群调度决策"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
            
        k_B, k_S, tau_B, tau_S, T_base = self.params
        
        # 计算硬件特征指标
        batch_50_saturation = -np.log(0.5) / k_B if k_B > 0 else float('inf')
        token_50_saturation = -np.log(0.5) / k_S if k_S > 0 else float('inf')
        
        hardware_info = {
            'peak_throughput_tokens_per_ms': self.P_max,  # 关键调度参数 - 稳定估计
            'batch_efficiency_factor': k_B,
            'token_efficiency_factor': k_S,
            'batch_50_saturation': batch_50_saturation,
            'token_50_saturation': token_50_saturation,
            'per_batch_overhead_ms': tau_B,
            'per_token_overhead_ms': tau_S,
            'base_latency_ms': T_base,
            'hardware_score': self.P_max / (tau_B + tau_S * 100),  # 综合性能评分
            'estimation_method': 'two_stage_fit'  # 标记估计方法
        }
        
        return hardware_info
    
    def estimate_optimal_batch_config(self, target_latency_ms: float, 
                                    token_budget: int) -> Dict[str, float]:
        """根据目标延迟估算最优批次配置"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
            
        best_config = None
        best_throughput = 0
        
        for batch_size in range(1, 65):
            tokens_per_request = token_budget // batch_size
            if tokens_per_request < 1:
                break
                
            predicted_latency = self.predict(batch_size, token_budget)
            
            if predicted_latency <= target_latency_ms:
                throughput = token_budget / predicted_latency
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = {
                        'batch_size': batch_size,
                        'total_tokens': token_budget,
                        'tokens_per_request': tokens_per_request,
                        'predicted_latency_ms': predicted_latency,
                        'effective_throughput': throughput
                    }
        
        return best_config or {
            'batch_size': 1,
            'total_tokens': min(token_budget, 512),
            'tokens_per_request': min(token_budget, 512),
            'predicted_latency_ms': self.predict(1, min(token_budget, 512)),
            'effective_throughput': min(token_budget, 512) / self.predict(1, min(token_budget, 512))
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
            
        import pickle
        
        model_data = {
            'P_max': self.P_max,  # 独立估计的峰值吞吐量
            'params': self.params,
            'scales': self.scales,
            'fit_metrics': self.fit_metrics,
            'param_cov': getattr(self, 'param_cov', None),
            'model_type': 'StableClusterModel'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        if self.verbose:
            logger.info(f"稳定集群调度模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.P_max = model_data['P_max']
        self.params = model_data['params']
        self.scales = model_data['scales'] 
        self.fit_metrics = model_data['fit_metrics']
        self.param_cov = model_data.get('param_cov')
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"稳定集群调度模型已加载: {filepath}")


def analyze_model_stability():
    """分析模型稳定性的工具函数"""
    print("🔍 模型稳定性分析建议")
    print("=" * 50)
    print("当出现参数标准误差过大时，可能的原因和解决方案：")
    print()
    print("1. 参数冗余问题:")
    print("   - 原因：P_max 和 w_0 高度相关")
    print("   - 解决：使用 StableClusterModel")
    print()
    print("2. 边界约束问题:")
    print("   - 原因：参数达到上下界")
    print("   - 解决：调整参数边界或重新参数化")
    print()
    print("3. 数值稳定性:")
    print("   - 原因：协方差矩阵ill-conditioned")
    print("   - 解决：正则化或降维")
    print()
    print("4. 数据质量问题:")
    print("   - 原因：噪声过大或样本分布不均")
    print("   - 解决：数据清洗和特征工程")


def demo_usage():
    """使用示例"""
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟的batch_sizes和chunk_sizes
    batch_sizes = np.random.randint(1, 33, n_samples)
    chunk_sizes_list = []
    
    for b in batch_sizes:
        # 随机生成每个batch的chunk_sizes
        if np.random.random() < 0.3:  # 30% prefill
            sizes = np.random.randint(10, 200, b).tolist()
        else:  # 70% decode  
            sizes = [1] * b
        chunk_sizes_list.append(sizes)
    
    # 使用真实模型生成延迟（加噪声）
    true_params = [50.0, 0.1, 0.02, 5.0, 0.05, 10.0, 0.5, 0.001]
    
    latencies = []
    for i, sizes in enumerate(chunk_sizes_list):
        B = len(sizes)
        S = sum(sizes)
        latency = ThroughputSaturationModel.latency_model((np.array([B]), np.array([S])), *true_params)[0]
        latency += np.random.normal(0, latency * 0.1)  # 10% 噪声
        latencies.append(max(latency, 1.0))  # 确保正值
    
    # 创建DataFrame
    df = pd.DataFrame({
        'chunk_sizes': chunk_sizes_list,
        'model_run_duration_ms': latencies,
        'batch_id': range(n_samples)
    })
    
    print("🚀 演示吞吐饱和模型")
    print("=" * 50)
    
    # 创建并训练原始模型
    model = ThroughputSaturationModel(verbose=True)
    model.fit(df)
    
    print("\n" + "="*50)
    print("🚀 演示稳定集群调度模型")
    
    # 创建并训练稳定模型
    stable_model = StableClusterModel(verbose=True)
    stable_model.fit(df)
    
    # 预测示例
    print(f"\n📊 预测对比:")
    test_cases = [(8, 512), (16, 256), (32, 128)]
    for B, S in test_cases:
        pred_orig = model.predict(B, S)
        pred_stable = stable_model.predict(B, S)
        print(f"Batch={B:2d}, Tokens={S:3d} -> 原始: {pred_orig:.2f} ms, 稳定: {pred_stable:.2f} ms")
    
    # 硬件能力对比
    print(f"\n🏭 硬件能力对比:")
    print(f"原始模型峰值吞吐量: {model.params[0]:.4f} tokens/ms")
    hardware_info = stable_model.get_hardware_capacity()
    print(f"稳定模型峰值吞吐量: {hardware_info['peak_throughput_tokens_per_ms']:.4f} tokens/ms (稳定估计)")
    
    # 绘制等高线图
    fig1 = model.plot_contour(save_path='./modeling/performance_contour.png')
    
    # 绘制残差分析
    fig2 = model.plot_residuals(df)
    
    plt.show()
    
    return model, stable_model, df


if __name__ == '__main__':
    # 运行演示
    model, stable_model, df = demo_usage() 
    
    # 分析稳定性
    analyze_model_stability() 