#!/usr/bin/env python3
"""
基于吞吐饱和理论的性能建模模块

该模块实现了论文中描述的物理可解释的性能预测模型，用于SLA感知调度器。
包含完整的模型训练、预测和序列化功能。
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from typing import Tuple, Dict, Optional, Union
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ThroughputSaturationModel:
    """基于吞吐饱和理论的性能建模类
    
    实现论文中的核心性能模型：
    Thr(B,S) = P_max * (1 - exp(-k_B * B)) * (1 - exp(-k_S * S))
    Work(B,S) = w_0 + w_1 * S
    T(B,S) = τ_0 + Work(B,S) / Thr(B,S) + τ_B * B + τ_S * S
    """
    
    def __init__(self, verbose: bool = True):
        """初始化模型
        
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
        """计算有效吞吐量"""
        return P_max * (1.0 - np.exp(-k_B * B)) * (1.0 - np.exp(-k_S * S))
    
    @staticmethod
    def workload(S: np.ndarray, w_0: float, w_1: float) -> np.ndarray:
        """计算工作量"""
        return w_0 + w_1 * S
    
    @staticmethod
    def latency_model(X: Tuple[np.ndarray, np.ndarray], 
                     P_max: float, k_B: float, k_S: float, 
                     w_0: float, w_1: float, 
                     tau_0: float, tau_B: float, tau_S: float) -> np.ndarray:
        """完整的延迟模型"""
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
        """从DataFrame中提取特征"""
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
        """数据预处理：提取特征、过滤异常值、检查有效性"""
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
            (T < 300)  # 过滤过长延迟
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
        """特征归一化，使用中位数缩放"""
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
        """参数初始化"""
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
        upper_bounds = [1e6, 10.0, 10.0, 1e3, 1e2, 1e3, 1e1, 1e1]
        
        if self.verbose:
            logger.info(f"参数初始化: P_max={P_max_init:.3f}, w_1={w_1_init:.6f}")
            
        return p0, lower_bounds, upper_bounds
    
    def fit(self, df: pd.DataFrame) -> 'ThroughputSaturationModel':
        """拟合模型"""
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
                    method='trf'
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
    
    def predict(self, batch_size: Union[float, np.ndarray], 
                total_tokens: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """预测延迟"""
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
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要信息"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        summary = {
            'parameters': dict(zip(self.param_names, self.params)),
            'metrics': self.fit_metrics,
            'scales': {'batch_scale': self.scales[0], 'token_scale': self.scales[1]}
        }
        
        return summary
    
    def save_model(self, filepath: str):
        """保存模型到文件"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用 fit() 方法")
            
        model_data = {
            'params': self.params,
            'scales': self.scales,
            'fit_metrics': self.fit_metrics,
            'param_cov': getattr(self, 'param_cov', None),
            'param_names': self.param_names,
            'param_descriptions': self.param_descriptions
        }
        
        # 确保目录存在
        dirname = os.path.dirname(filepath)
        if dirname:  # 只有当目录路径不为空时才创建
            os.makedirs(dirname, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        if self.verbose:
            logger.info(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """从文件加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.params = model_data['params']
        self.scales = model_data['scales'] 
        self.fit_metrics = model_data['fit_metrics']
        self.param_cov = model_data.get('param_cov')
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"Model loaded: {filepath}")
            logger.info(f"Model R²: {self.fit_metrics['r2']:.4f}")
    
    @classmethod
    def load_from_file(cls, filepath: str, verbose: bool = True) -> 'ThroughputSaturationModel':
        """类方法：从文件加载模型"""
        model = cls(verbose=verbose)
        model.load_model(filepath)
        return model
