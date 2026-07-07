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
    """基于吞吐饱和理论的性能建模类 (6-parameter model)

    实现论文中的核心性能模型：
    Thr(B,S) = P_max * (1 - exp(-k_B * B)) * (1 - exp(-k_S * S))
    T(B,S) = w_1 * S / Thr(B,S) + τ_B * B + τ_S * S

    使用两阶段拟合解耦结构参数 (P_max, k_B, k_S) 与线性参数 (w_1, τ_B, τ_S)。
    """

    def __init__(self, verbose: bool = True):
        """初始化模型

        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.is_fitted = False
        self.P_max = None
        self.params = None
        self.scales = None
        self.fit_metrics = None

        # 参数名称和含义 (6 parameters)
        self.param_names = [
            'P_max', 'k_B', 'k_S', 'w_1', 'tau_B', 'tau_S'
        ]
        self.param_descriptions = {
            'P_max': '最大有效吞吐量 (tokens/ms)',
            'k_B': 'batch并行度敏感系数',
            'k_S': 'token并行度敏感系数',
            'w_1': '每token工作量系数',
            'tau_B': '每batch额外延迟 (ms/batch)',
            'tau_S': '每token额外延迟 (ms/token)'
        }
    
    @staticmethod
    def throughput(B: np.ndarray, S: np.ndarray, P_max: float, k_B: float, k_S: float) -> np.ndarray:
        """计算有效吞吐量"""
        return P_max * (1.0 - np.exp(-k_B * B)) * (1.0 - np.exp(-k_S * S))

    @staticmethod
    def latency_model(X: Tuple[np.ndarray, np.ndarray],
                     P_max: float, k_B: float, k_S: float,
                     w_1: float, tau_B: float, tau_S: float) -> np.ndarray:
        """完整的延迟模型 (6 parameters)"""
        B, S = X
        thr = P_max * (1.0 - np.exp(-k_B * B)) * (1.0 - np.exp(-k_S * S))
        thr = np.maximum(thr, 1e-9)
        return w_1 * S / thr + tau_B * B + tau_S * S
    
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
        if 'chunk_sizes' in df.columns:
            B = df['chunk_sizes'].apply(_compute_batch_size).to_numpy(dtype=float)
            S = df['chunk_sizes'].apply(_compute_total_tokens).to_numpy(dtype=float)
        else:
            B = df['batch_size'].to_numpy(dtype=float)
            S = df['total_tokens'].to_numpy(dtype=float)
        T = df['model_run_duration_ms'].to_numpy(dtype=float)
        
        return B, S, T
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据预处理：提取特征、过滤异常值、检查有效性"""
        if self.verbose:
            logger.info("Starting data preprocessing...")
            
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
            logger.info(f"Data filtering: {initial_count} -> {len(T)} samples")
            logger.info(f"Batch Size range: [{B.min():.1f}, {B.max():.1f}]")
            logger.info(f"Total Tokens range: [{S.min():.1f}, {S.max():.1f}]")
            logger.info(f"Latency range: [{T.min():.2f}, {T.max():.2f}] ms")
        
        if len(T) < 20:
            raise ValueError(f"Insufficient valid samples: {len(T)}, recommend at least 20 samples")
            
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
            logger.info(f"Feature normalization: B_scale={B_scale:.2f}, S_scale={S_scale:.2f}")
            
        return B_norm, S_norm, scales
    
    def _initialize_parameters(self, B: np.ndarray, S: np.ndarray, T: np.ndarray) -> Tuple[list, list, list]:
        """参数初始化 (6 parameters)"""
        # 估计峰值吞吐量
        throughput_estimates = S / np.maximum(T, 1e-6)
        P_max_init = max(np.percentile(throughput_estimates, 95), 1e-3)

        # 线性拟合估计w_1
        try:
            w_1_init = max(np.polyfit(S, T, 1)[0], 1e-9)
        except Exception:
            w_1_init = 0.01

        # 初始参数: [P_max, k_B, k_S, w_1, tau_B, tau_S]
        p0 = [
            P_max_init,     # P_max
            0.1,            # k_B
            0.01,           # k_S
            w_1_init,       # w_1
            1e-3,           # tau_B
            1e-3            # tau_S
        ]

        # 参数下界（物理合理性）
        lower_bounds = [1e-6, 1e-6, 1e-6, 1e-9, 0.0, 0.0]

        # 参数上界（防止过拟合）
        upper_bounds = [1e6, 10.0, 10.0, 1e2, 1e1, 1e1]

        if self.verbose:
            logger.info(f"Parameter initialization: P_max={P_max_init:.3f}, w_1={w_1_init:.6f}")

        return p0, lower_bounds, upper_bounds
    
    def _estimate_P_max(self, B: np.ndarray, S: np.ndarray, T: np.ndarray) -> float:
        """Data-driven estimation of peak throughput P_max.

        Uses the top-percentile effective throughput from large-batch samples
        to anchor P_max, preventing it from drifting due to identifiability
        issues with saturation parameters k_B, k_S.
        """
        large_mask = (B >= np.percentile(B, 75)) & (S >= np.percentile(S, 75))
        if np.sum(large_mask) < 10:
            eff_thr = S / np.maximum(T, 1e-6)
        else:
            eff_thr = S[large_mask] / np.maximum(T[large_mask], 1e-6)
        return float(np.percentile(eff_thr, 95))

    def fit(self, df: pd.DataFrame) -> 'ThroughputSaturationModel':
        """两阶段拟合模型

        Stage 1: 数据驱动估计 P_max (固定) + 线性回归估计 τ_B, τ_S
        Stage 2: 固定 P_max, τ_B, τ_S，用 curve_fit 拟合 k_B, k_S, w_1
        Stage 3: 固定 P_max, k_B, k_S，重新拟合 w_1, τ_B, τ_S
        """
        if self.verbose:
            logger.info("Starting two-stage model fitting...")

        # 数据预处理
        B, S, T = self._preprocess_data(df)

        # 特征归一化
        B_norm, S_norm, scales = self._normalize_features(B, S)
        self.scales = scales

        # 参数初始化
        p0, lower_bounds, upper_bounds = self._initialize_parameters(B_norm, S_norm, T)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # === Stage 1: 固定 P_max + 线性回归估计 τ_B, τ_S ===
                P_max_fix = self._estimate_P_max(B_norm, S_norm, T)

                # T ≈ α + τ_B·B + τ_S·S
                A_lin = np.column_stack([np.ones(len(T)), B_norm, S_norm])
                lin_result, _, _, _ = np.linalg.lstsq(A_lin, T, rcond=None)
                tau_B_init = max(lin_result[1], 0.0)
                tau_S_init = max(lin_result[2], 0.0)

                if self.verbose:
                    logger.info(f"Stage 1: P_max={P_max_fix:.4f}, "
                                f"tau_B={tau_B_init:.4f}, tau_S={tau_S_init:.4f}")

                # === Stage 2: 固定 P_max, τ_B, τ_S，拟合 k_B, k_S, w_1 ===
                T_residual = T - tau_B_init * B_norm - tau_S_init * S_norm
                T_residual = np.maximum(T_residual, 1e-6)

                def stage2_model(X, k_B, k_S, w_1):
                    B, S = X
                    thr = P_max_fix * (1.0 - np.exp(-k_B * B)) * (1.0 - np.exp(-k_S * S))
                    thr = np.maximum(thr, 1e-9)
                    return w_1 * S / thr

                p0_s2 = [p0[1], p0[2], p0[3]]  # k_B, k_S, w_1
                lb_s2 = [lower_bounds[1], lower_bounds[2], lower_bounds[3]]
                ub_s2 = [upper_bounds[1], upper_bounds[2], upper_bounds[3]]

                popt_s2, _ = curve_fit(
                    stage2_model,
                    xdata=(B_norm, S_norm),
                    ydata=T_residual,
                    p0=p0_s2,
                    bounds=(lb_s2, ub_s2),
                    maxfev=20000,
                    method='trf'
                )
                k_B_fix, k_S_fix, w_1_s2 = popt_s2

                if self.verbose:
                    logger.info(f"Stage 2 (structural): k_B={k_B_fix:.4f}, "
                                f"k_S={k_S_fix:.4f}, w_1={w_1_s2:.6f}")

                # === Stage 3: 固定 P_max, k_B, k_S，重新拟合 w_1, τ_B, τ_S ===
                def stage3_model(X, w_1, tau_B, tau_S):
                    B, S = X
                    thr = P_max_fix * (1.0 - np.exp(-k_B_fix * B)) * (1.0 - np.exp(-k_S_fix * S))
                    thr = np.maximum(thr, 1e-9)
                    return w_1 * S / thr + tau_B * B + tau_S * S

                p0_s3 = [w_1_s2, tau_B_init, tau_S_init]
                lb_s3 = [lower_bounds[3], lower_bounds[4], lower_bounds[5]]
                ub_s3 = [upper_bounds[3], upper_bounds[4], upper_bounds[5]]

                popt_s3, pcov_s3 = curve_fit(
                    stage3_model,
                    xdata=(B_norm, S_norm),
                    ydata=T,
                    p0=p0_s3,
                    bounds=(lb_s3, ub_s3),
                    maxfev=20000,
                    method='trf'
                )
                w_1_final, tau_B_final, tau_S_final = popt_s3

                if self.verbose:
                    logger.info(f"Stage 3 (linear): w_1={w_1_final:.6f}, "
                                f"tau_B={tau_B_final:.4f}, tau_S={tau_S_final:.4f}")

            # 合并 6 个参数: [P_max, k_B, k_S, w_1, tau_B, tau_S]
            self.params = np.array([P_max_fix, k_B_fix, k_S_fix,
                                    w_1_final, tau_B_final, tau_S_final])
            self.param_cov = pcov_s3  # 最后一步的协方差
            self.is_fitted = True
            self.P_max = P_max_fix

            # 计算拟合质量指标
            T_pred = self.latency_model((B_norm, S_norm), *self.params)
            self.fit_metrics = {
                'r2': r2_score(T, T_pred),
                'rmse': np.sqrt(mean_squared_error(T, T_pred)),
                'mae': mean_absolute_error(T, T_pred),
                'n_samples': len(T)
            }

            if self.verbose:
                self._print_fit_results()

        except Exception as e:
            logger.error(f"Fitting failed: {e}")
            raise

        return self
    
    def _print_fit_results(self):
        """打印拟合结果"""
        logger.info("Fitting completed!")
        logger.info(f"R² = {self.fit_metrics['r2']:.4f}")
        logger.info(f"RMSE = {self.fit_metrics['rmse']:.3f} ms")
        logger.info(f"MAE = {self.fit_metrics['mae']:.3f} ms")
        logger.info(f"Sample count = {self.fit_metrics['n_samples']}")
        
        logger.info("\nFitted parameters:")
        for i, (name, desc) in enumerate(zip(self.param_names, self.param_descriptions.values())):
            logger.info(f"{name:8s} = {self.params[i]:10.6f}  # {desc}")
    
    def predict(self, batch_size: Union[float, np.ndarray], 
                total_tokens: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """预测延迟"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet, please call fit() method first")
            
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
            raise ValueError("Model not fitted yet, please call fit() method first")
            
        summary = {
            'parameters': dict(zip(self.param_names, self.params)),
            'metrics': self.fit_metrics,
            'scales': {'batch_scale': self.scales[0], 'token_scale': self.scales[1]}
        }
        
        return summary
    
    def save_model(self, filepath: str):
        """保存模型到文件"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet, please call fit() method first")
            
        model_data = {
            'P_max': self.P_max,
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
        
        try:
            self.P_max = model_data['P_max']
        except:
            self.P_max = model_data['params'][0]
        self.params = model_data['params']
        self.scales = model_data['scales'] 
        self.fit_metrics = model_data['fit_metrics']
        self.param_cov = model_data.get('param_cov')
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"Model loaded: {filepath}")
            logger.info(f"Model R²: {self.fit_metrics['r2']:.4f}, P_max: {self.P_max:.6f} tokens/ms")
    
    @classmethod
    def load_from_file(cls, filepath: str, verbose: bool = True) -> 'ThroughputSaturationModel':
        """类方法：从文件加载模型"""
        model = cls(verbose=verbose)
        model.load_model(filepath)
        return model

    def partial_fit(self, df: pd.DataFrame) -> 'ThroughputSaturationModel':
        """Partial fit: fix structural params (P_max, k_B, k_S), only fit linear params (w_1, tau_B, tau_S).

        Requires a pretrained model to be loaded first (self.params and self.scales must exist).
        This implements the multi-timescale adaptation: structural parameters from offline profiling
        are treated as slow-changing hardware properties, while linear parameters adapt online.
        """
        if self.params is None or self.scales is None:
            raise ValueError("partial_fit requires a pretrained model to be loaded first")

        if self.verbose:
            logger.info("Starting partial fit (fixing P_max, k_B, k_S)...")

        # Data preprocessing
        B, S, T = self._preprocess_data(df)

        # Use scales from the pretrained model
        B_scale, S_scale = self.scales
        B_norm = B / B_scale
        S_norm = S / S_scale

        # Fix structural params from pretrained model
        P_max_fix = self.params[0]
        k_B_fix = self.params[1]
        k_S_fix = self.params[2]

        if self.verbose:
            logger.info(f"Fixed structural params: P_max={P_max_fix:.4f}, "
                        f"k_B={k_B_fix:.4f}, k_S={k_S_fix:.6f}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Fit only w_1, tau_B, tau_S with structural params fixed
                def partial_model(X, w_1, tau_B, tau_S):
                    B, S = X
                    thr = P_max_fix * (1.0 - np.exp(-k_B_fix * B)) * (1.0 - np.exp(-k_S_fix * S))
                    thr = np.maximum(thr, 1e-9)
                    return w_1 * S / thr + tau_B * B + tau_S * S

                # Initial values from current params (or sensible defaults)
                w_1_init = self.params[3] if self.params[3] > 0 else 0.01
                tau_B_init = self.params[4] if self.params[4] > 0 else 1e-3
                tau_S_init = self.params[5] if self.params[5] > 0 else 1e-3

                p0 = [w_1_init, tau_B_init, tau_S_init]
                lb = [1e-9, 0.0, 0.0]
                ub = [1e2, 1e1, 1e1]

                popt, pcov = curve_fit(
                    partial_model,
                    xdata=(B_norm, S_norm),
                    ydata=T,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=20000,
                    method='trf'
                )
                w_1_fit, tau_B_fit, tau_S_fit = popt

                if self.verbose:
                    logger.info(f"Partial fit result: w_1={w_1_fit:.6f}, "
                                f"tau_B={tau_B_fit:.4f}, tau_S={tau_S_fit:.4f}")

            # Update only linear params, keep structural params fixed
            self.params = np.array([P_max_fix, k_B_fix, k_S_fix,
                                    w_1_fit, tau_B_fit, tau_S_fit])
            self.param_cov = pcov
            self.is_fitted = True
            self.P_max = P_max_fix

            # Compute fit metrics
            T_pred = self.latency_model((B_norm, S_norm), *self.params)
            self.fit_metrics = {
                'r2': r2_score(T, T_pred),
                'rmse': np.sqrt(mean_squared_error(T, T_pred)),
                'mae': mean_absolute_error(T, T_pred),
                'n_samples': len(T)
            }

            if self.verbose:
                self._print_fit_results()

        except Exception as e:
            logger.error(f"Partial fit failed: {e}")
            raise

        return self

    def get_model_parameters(self) -> Dict:
        """获取模型参数"""
        return {
            'P_max': self.P_max,
            'params': self.params,
            'scales': self.scales,
            'fit_metrics': self.fit_metrics,
            'param_cov': getattr(self, 'param_cov', None),
            'param_names': self.param_names,
            'param_descriptions': self.param_descriptions
        }


class StableClusterModel:
    """稳定的集群调度性能模型
    
    使用两阶段拟合方法确保P_max参数稳定性，专为异构集群调度设计。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.is_fitted = False
        self.P_max = None
        self.params = None
        self.scales = None
        self.fit_metrics = None
        
        self.param_names = ['k_B', 'k_S', 'tau_B', 'tau_S', 'T_base']
    
    def _estimate_peak_throughput(self, B: np.ndarray, S: np.ndarray, T: np.ndarray) -> float:
        """独立估计峰值吞吐量P_max"""
        large_batch_mask = (B >= np.percentile(B, 80)) & (S >= np.percentile(S, 80))
        
        if np.sum(large_batch_mask) < 10:
            effective_throughput = S / T
        else:
            effective_throughput = S[large_batch_mask] / T[large_batch_mask]
        
        P_max_estimate = np.percentile(effective_throughput, 90)
        
        if self.verbose:
            logger.info(f"Peak throughput estimation: {P_max_estimate:.4f} tokens/ms")
            
        return P_max_estimate
    
    @staticmethod
    def stable_latency_model(X: Tuple[np.ndarray, np.ndarray], 
                           k_B: float, k_S: float, tau_B: float, tau_S: float, T_base: float,
                           P_max_fixed: float) -> np.ndarray:
        """稳定的延迟模型，P_max作为固定参数传入"""
        B, S = X
        
        eff_B = 1.0 - np.exp(-k_B * B)
        eff_S = 1.0 - np.exp(-k_S * S)
        
        eff_B = np.maximum(eff_B, 1e-6)
        eff_S = np.maximum(eff_S, 1e-6)
        
        latency = T_base / (eff_B * eff_S) + tau_B * B + tau_S * S
        
        return latency
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从DataFrame中提取特征"""
        # 复用ThroughputSaturationModel的特征提取逻辑
        temp_model = ThroughputSaturationModel(verbose=False)
        return temp_model._extract_features(df)
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据预处理"""
        temp_model = ThroughputSaturationModel(verbose=self.verbose)
        return temp_model._preprocess_data(df)
    
    def _normalize_features(self, B: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """特征归一化"""
        temp_model = ThroughputSaturationModel(verbose=self.verbose)
        return temp_model._normalize_features(B, S)
    
    def fit(self, df: pd.DataFrame) -> 'StableClusterModel':
        """两阶段拟合方法：1.估计P_max 2.拟合其他参数"""
        if self.verbose:
            logger.info("Starting stable cluster scheduling model fitting...")
            
        B, S, T = self._preprocess_data(df)
        B_norm, S_norm, scales = self._normalize_features(B, S)
        self.scales = scales
        
        # 阶段1：独立估计P_max
        self.P_max = self._estimate_peak_throughput(B_norm, S_norm, T)
        
        # 阶段2：固定P_max，拟合其他参数
        T_median = np.median(T)
        p0 = [0.5, 0.2, 1.0, 0.1, T_median * 0.5]
        lower_bounds = [0.01, 0.001, 0.0, 0.0, 0.1]
        upper_bounds = [5.0, 2.0, 20.0, 2.0, 200.0]
        
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
            logger.error(f"Stable cluster scheduling model fitting failed: {e}")
            raise
            
        return self
    
    def _print_fit_results(self):
        """打印拟合结果"""
        logger.info("Stable cluster scheduling model fitting completed!")
        logger.info(f"R² = {self.fit_metrics['r2']:.4f}")
        logger.info(f"RMSE = {self.fit_metrics['rmse']:.3f} ms")
        logger.info(f"MAE = {self.fit_metrics['mae']:.3f} ms")
        logger.info(f"Sample count = {self.fit_metrics['n_samples']}")
        logger.info(f"P_max = {self.P_max:.6f} tokens/ms (stable estimation)")
    
    def predict(self, batch_size: Union[float, np.ndarray], 
                total_tokens: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """预测延迟"""
        if not self.is_fitted:
            raise ValueError("Stable cluster model not fitted yet")
            
        B = np.asarray(batch_size)
        S = np.asarray(total_tokens)
        
        B_scale, S_scale = self.scales
        B_norm = B / B_scale
        S_norm = S / S_scale
        
        latency = self.stable_latency_model((B_norm, S_norm), *self.params, self.P_max)
        
        return latency
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要信息"""
        if not self.is_fitted:
            raise ValueError("Stable cluster model not fitted yet")
            
        return {
            'parameters': {'P_max': self.P_max, **dict(zip(self.param_names, self.params))},
            'metrics': self.fit_metrics,
            'scales': {'batch_scale': self.scales[0], 'token_scale': self.scales[1]},
            'model_type': 'StableClusterModel'
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Stable cluster model not fitted yet")
            
        model_data = {
            'P_max': self.P_max,
            'params': self.params,
            'scales': self.scales,
            'fit_metrics': self.fit_metrics,
            'param_names': self.param_names,
            'model_type': 'StableClusterModel'
        }
        
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        if self.verbose:
            logger.info(f"Stable cluster scheduling model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stable cluster model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.P_max = model_data['P_max']
        self.params = model_data['params']
        self.scales = model_data['scales'] 
        self.fit_metrics = model_data['fit_metrics']
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"Stable cluster scheduling model loaded: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str, verbose: bool = True) -> 'StableClusterModel':
        """类方法：从文件加载稳定集群模型"""
        model = cls(verbose=verbose)
        model.load_model(filepath)
        return model