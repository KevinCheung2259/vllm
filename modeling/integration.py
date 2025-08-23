#!/usr/bin/env python3
"""
与 scheduler_profiling_example.py 的集成模块
提供简化的接口来使用吞吐饱和建模功能
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 添加modeling目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from performance_model import ThroughputSaturationModel, StableClusterModel


def analyze_with_modeling(log_file_or_dir='profiling_result', use_stable_model=False, suffix='', save_outputs=True):
    """
    使用吞吐饱和模型分析profiling数据
    
    Args:
        log_file_or_dir: 日志文件或目录路径
        use_stable_model: 是否使用稳定集群模型
        suffix: 文件名后缀，用于区分不同的分析结果
        save_outputs: 是否保存模型和图片到文件系统
        
    Returns:
        (model, df) 元组
    """
    model_type = "稳定集群调度模型" if use_stable_model else "吞吐饱和模型"
    print(f"🔬 使用{model_type}分析profiling数据")
    print("=" * 50)
    
    # 读取数据
    df = read_profiling_data(log_file_or_dir)
    if df is None or df.empty:
        print("❌ 未找到有效数据")
        return None, None
    
    print(f"✅ 成功读取 {len(df)} 条profiling数据")
    
    # 创建并拟合模型
    try:
        if use_stable_model:
            model = StableClusterModel(verbose=True)
        else:
            model = ThroughputSaturationModel(verbose=True)
            
        model.fit(df)
        
        # 打印模型摘要
        if use_stable_model:
            print_stable_model_insights(model)
        else:
            print_model_insights(model)
        
        # 生成可视化（如果启用保存）
        if save_outputs:
            if use_stable_model:
                create_stable_model_plots(model, df, suffix)
            else:
                create_modeling_plots(model, df, suffix)

        # 测试模型性能
        test_model_performance(model)
        
        return model, df
        
    except Exception as e:
        print(f"❌ 建模失败: {e}")
        import traceback
        traceback.print_exc()
        return None, df


def test_model_performance(model):
    """测试模型性能"""
    import time
    
    test_points = [(3, 256), (16, 2048), (32, 4096)]
    print("\n🔍 模型性能测试:")
    print("-" * 40)
    
    for B, S in test_points:
        start_time = time.perf_counter()
        pred = model.predict(B, S)
        end_time = time.perf_counter()
        predict_cost = (end_time - start_time) * 1000  # 转换为毫秒
        
        print(f"Batch={B:2d}, Tokens={S:4d} -> {pred:.2f} ms (预测耗时: {predict_cost:.3f} ms)")


def read_profiling_data(log_file_or_dir):
    """读取profiling数据（复用原脚本逻辑）"""
    import json
    from pathlib import Path
    
    log_path = Path(log_file_or_dir)
    
    # 查找jsonl文件
    if log_path.is_dir():
        jsonl_files = list(log_path.glob('*.jsonl'))
        if not jsonl_files:
            print(f"❌ 在目录 {log_file_or_dir} 中没有找到jsonl文件")
            return None
        log_files = jsonl_files
        print(f"📁 找到目录: {log_file_or_dir}")
        print(f"📄 使用文件: {len(jsonl_files)} 个")
    else:
        if not log_path.exists():
            print(f"❌ 日志文件 {log_path} 不存在")
            return None
        log_files = [log_path]
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
    data = [item for item in data if item.get('schedule_duration_ms', 0) < 200]
    data = [item for item in data if item.get('model_run_duration_ms', 0) < 200]
    
    df = pd.DataFrame(data)
    
    # 计算decode和prefill请求数（复用原脚本逻辑）
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


def print_model_insights(model):
    """打印模型深度分析"""
    print("\n🔍 模型深度分析")
    print("=" * 50)
    
    params = model.params
    param_names = model.param_names
    
    # 系统特性分析
    P_max, k_B, k_S = params[0], params[1], params[2]
    w_0, w_1 = params[3], params[4]
    tau_0, tau_B, tau_S = params[5], params[6], params[7]
    
    print(f"🚀 系统吞吐特性:")
    print(f"   峰值吞吐量: {P_max:.2f} tokens/ms")
    print(f"   达到50%饱和的batch_size: {-np.log(0.5)/k_B:.1f}")
    print(f"   达到50%饱和的token数: {-np.log(0.5)/k_S:.0f}")
    
    print(f"\n⚡ 工作量特性:")
    print(f"   基础工作量: {w_0:.3f}")
    print(f"   每token工作量: {w_1:.6f}")
    if w_0 > 0:
        print(f"   固定开销对应token数: {w_0/w_1:.0f}")
    
    print(f"\n🕒 延迟构成:")
    print(f"   基础延迟: {tau_0:.2f} ms")
    print(f"   每batch开销: {tau_B:.3f} ms")
    print(f"   每token开销: {tau_S:.6f} ms")
    
    # 典型场景预测
    print(f"\n📊 典型场景预测:")
    scenarios = [
        ("小decode批次", 4, 4),
        ("中decode批次", 16, 16), 
        ("大decode批次", 64, 64),
        ("小prefill", 2, 128),
        ("大prefill", 1, 1024)
    ]
    
    for name, B, S in scenarios:
        latency = model.predict(B, S)
        print(f"   {name:12s}: B={B:2d}, S={S:4d} -> {latency:.1f} ms")


def create_modeling_plots(model, df, suffix=''):
    """创建建模相关的可视化图表"""
    print("\n📊 生成建模可视化图表...")
    
    # 构建文件名
    suffix_str = f"_{suffix}" if suffix else ""
    
    # 创建等高线图
    fig1 = model.plot_contour(
        batch_range=(1, 64),
        token_range=(1, 1024),
        save_path=f'./modeling/modeling_contour{suffix_str}.png'
    )
    
    # 创建残差分析图
    fig2 = model.plot_residuals(df)
    plt.savefig(f'./modeling/modeling_residuals{suffix_str}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Residual analysis plot saved: ./modeling/modeling_residuals{suffix_str}.png")
    
    # Create performance surface plot
    create_3d_surface(model, suffix)
    
    plt.show()





def create_3d_surface(model, suffix=''):
    """创建3D性能曲面图"""
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    # 构建文件名
    suffix_str = f"_{suffix}" if suffix else ""
    
    # 创建网格
    B_range = np.linspace(1, 32, 30)
    S_range = np.linspace(1, 512, 30)
    B_mesh, S_mesh = np.meshgrid(B_range, S_range)
    
    # 预测延迟
    T_mesh = model.predict(B_mesh.flatten(), S_mesh.flatten()).reshape(B_mesh.shape)
    
    # 绘制3D曲面
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(B_mesh, S_mesh, T_mesh, 
                          cmap='viridis', alpha=0.9, 
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Total Tokens')
    ax.set_zlabel('Latency (ms)')
    ax.set_title('vLLM Performance 3D Surface\n(Throughput Saturation Model)')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    plt.savefig(f'./modeling/modeling_3d_surface{suffix_str}.png', dpi=300, bbox_inches='tight')
    print(f"📊 3D performance surface plot saved: ./modeling/modeling_3d_surface{suffix_str}.png")





def compare_models(df, suffix='', save_outputs=True):
    """比较不同模型的性能"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    
    print("\n📈 模型对比分析")
    print("=" * 50)
    
    # 提取特征
    B = df['chunk_sizes'].apply(lambda x: len(x) if isinstance(x, list) else np.nan)
    S = df['chunk_sizes'].apply(lambda x: sum(x) if isinstance(x, list) else x)
    T = df['model_run_duration_ms']
    
    # 过滤有效数据
    mask = B.notna() & S.notna() & T.notna() & (B > 0) & (S > 0) & (T > 0)
    B, S, T = B[mask].values, S[mask].values, T[mask].values
    
    models = {}
    
    # 1. Linear model (S only)
    lr_s = LinearRegression()
    lr_s.fit(S.reshape(-1, 1), T)
    pred_lr_s = lr_s.predict(S.reshape(-1, 1))
    models['Linear(S)'] = {'pred': pred_lr_s, 'r2': lr_s.score(S.reshape(-1, 1), T)}
    
    # 2. Linear model (B+S)
    lr_bs = LinearRegression()
    X_bs = np.column_stack([B, S])
    lr_bs.fit(X_bs, T)
    pred_lr_bs = lr_bs.predict(X_bs)
    models['Linear(B+S)'] = {'pred': pred_lr_bs, 'r2': lr_bs.score(X_bs, T)}
    
    # 3. Polynomial model
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X_bs)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, T)
    pred_poly = lr_poly.predict(X_poly)
    models['Polynomial'] = {'pred': pred_poly, 'r2': lr_poly.score(X_poly, T)}
    
    # 4. Throughput saturation model
    ts_model = ThroughputSaturationModel(verbose=False)
    ts_model.fit(df)
    B_norm, S_norm, _ = ts_model._normalize_features(B, S)
    pred_ts = ts_model.latency_model((B_norm, S_norm), *ts_model.params)
    models['Throughput Saturation'] = {'pred': pred_ts, 'r2': ts_model.fit_metrics['r2']}
    
    # Print comparison results
    print("Model Performance Comparison (R²):")
    for name, result in models.items():
        print(f"  {name:20s}: {result['r2']:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(models.items()):
        ax = axes[i]
        ax.scatter(T, result['pred'], alpha=0.6, s=20)
        ax.plot([T.min(), T.max()], [T.min(), T.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Latency (ms)')
        ax.set_ylabel('Predicted Latency (ms)')
        ax.set_title(f'{name} (R²={result["r2"]:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_outputs:
        plt.savefig(f'./modeling/model_comparison_{suffix}.png', dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison plot saved: ./modeling/model_comparison_{suffix}.png")
    
    plt.show()
    
    return models


def print_stable_model_insights(model):
    """打印稳定模型的深度分析"""
    print("\n🔍 稳定模型深度分析")
    print("=" * 50)
    
    hardware_info = model.get_hardware_capacity()
    
    # 系统特性分析
    P_max = hardware_info['peak_throughput_tokens_per_ms']
    k_B = hardware_info['batch_efficiency_factor']
    k_S = hardware_info['token_efficiency_factor']
    tau_B = hardware_info['per_batch_overhead_ms']
    tau_S = hardware_info['per_token_overhead_ms']
    T_base = hardware_info['base_latency_ms']
    
    print(f"🚀 系统吞吐特性:")
    print(f"   峰值吞吐量: {P_max:.4f} tokens/ms (稳定估计)")
    print(f"   达到50%饱和的batch_size: {hardware_info['batch_50_saturation']:.1f}")
    print(f"   达到50%饱和的token数: {hardware_info['token_50_saturation']:.1f}")
    
    print(f"\n⚡ 延迟构成:")
    print(f"   基础延迟: {T_base:.2f} ms")
    print(f"   每batch开销: {tau_B:.3f} ms")
    print(f"   每token开销: {tau_S:.6f} ms")
    
    print(f"\n🏭 硬件评分:")
    print(f"   综合性能评分: {hardware_info['hardware_score']:.4f}")
    print(f"   估计方法: {hardware_info['estimation_method']}")
    
    # 典型场景预测
    print(f"\n📊 典型场景预测:")
    scenarios = [
        ("小decode批次", 4, 4),
        ("中decode批次", 16, 16), 
        ("大decode批次", 64, 64),
        ("小prefill", 2, 128),
        ("大prefill", 1, 1024)
    ]
    
    for name, B, S in scenarios:
        latency = model.predict(B, S)
        print(f"   {name:12s}: B={B:2d}, S={S:4d} -> {latency:.1f} ms")


def create_stable_model_plots(model, df, suffix=''):
    """创建稳定模型相关的可视化图表"""
    print("\n📊 生成稳定模型可视化图表...")
    
    # 构建文件名
    suffix_str = f"_{suffix}" if suffix else ""
    
    # 创建等高线图 (复用原有函数，稳定模型也有predict方法)
    try:
        # 创建等高线图
        B_range = np.linspace(1, 64, 50)
        S_range = np.linspace(1, 1024, 50)
        B_mesh, S_mesh = np.meshgrid(B_range, S_range)
        
        # 预测延迟
        T_mesh = model.predict(B_mesh.flatten(), S_mesh.flatten()).reshape(B_mesh.shape)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 填充等高线
        cs1 = ax.contourf(B_mesh, S_mesh, T_mesh, levels=20, cmap='viridis', alpha=0.8)
        cbar1 = plt.colorbar(cs1, ax=ax)
        cbar1.set_label('Model Run Latency (ms)', fontsize=12)
        
        # 等高线
        cs2 = ax.contour(B_mesh, S_mesh, T_mesh, levels=10, colors='white', linewidths=1, alpha=0.7)
        ax.clabel(cs2, inline=True, fontsize=9, fmt='%1.0f ms')
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Total Tokens', fontsize=12)
        ax.set_title('vLLM Model Run Latency Prediction\n(Stable Cluster Model)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加模型信息
        if model.fit_metrics:
            info_text = f"R² = {model.fit_metrics['r2']:.3f}\nRMSE = {model.fit_metrics['rmse']:.1f} ms\nP_max = {model.P_max:.3f} tokens/ms"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'./modeling/stable_model_contour{suffix_str}.png', dpi=300, bbox_inches='tight')
        print(f"📊 Stable model contour plot saved: ./modeling/stable_model_contour{suffix_str}.png")
        
    except Exception as e:
        print(f"⚠️  等高线图创建失败: {e}")
    
    # 创建残差分析图
    try:
        create_stable_residuals_plot(model, df, suffix)
    except Exception as e:
        print(f"⚠️  残差图创建失败: {e}")





def create_stable_residuals_plot(model, df, suffix=''):
    """创建稳定模型的残差分析图"""
    # 构建文件名
    suffix_str = f"_{suffix}" if suffix else ""
    
    # 获取数据
    B, S, T = model._preprocess_data(df)
    B_norm, S_norm, _ = model._normalize_features(B, S)
    
    # 预测值
    T_pred = model.stable_latency_model((B_norm, S_norm), *model.params, model.P_max)
    residuals = T - T_pred
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 预测值 vs 真实值
    axes[0].scatter(T_pred, T, alpha=0.6, s=20)
    axes[0].plot([T.min(), T.max()], [T.min(), T.max()], 'r--', lw=2)
    axes[0].set_xlabel('Predicted Latency (ms)')
    axes[0].set_ylabel('Actual Latency (ms)')
    axes[0].set_title('Predicted vs Actual (Stable Model)')
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
    axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Residuals (ms)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residuals Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./modeling/stable_model_residuals{suffix_str}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Stable model residuals plot saved: ./modeling/stable_model_residuals{suffix_str}.png")


def analyze_cluster_scheduling(log_file_or_dir='profiling_result', suffix='', save_outputs=True):
    """
    专门用于集群调度的分析函数
    
    Args:
        log_file_or_dir: 日志文件或目录路径
        suffix: 文件名后缀，用于区分不同的分析结果
        save_outputs: 是否保存模型和图片到文件系统
        
    Returns:
        (model, hardware_info, df) 元组
    """
    print("🏭 集群调度性能分析")
    print("=" * 50)
    
    # 使用稳定模型进行分析
    model, df = analyze_with_modeling(log_file_or_dir, use_stable_model=True, suffix=suffix, save_outputs=save_outputs)
    
    if model is None:
        return None, None, df
    
    # 获取硬件能力信息
    hardware_info = model.get_hardware_capacity()
    
    print("\n🏭 硬件能力分析 (用于集群调度)")
    print("-" * 40)
    print("💡 关键调度参数:")
    print(f"  峰值吞吐量 (P_max): {hardware_info['peak_throughput_tokens_per_ms']:.4f} tokens/ms")
    print(f"  硬件性能评分: {hardware_info['hardware_score']:.4f}")
    print(f"  Batch 50%饱和点: {hardware_info['batch_50_saturation']:.2f}")
    print(f"  Token 50%饱和点: {hardware_info['token_50_saturation']:.2f}")
    
    print(f"\n📊 开销分析:")
    print(f"  基础延迟: {hardware_info['base_latency_ms']:.2f} ms")
    print(f"  每batch开销: {hardware_info['per_batch_overhead_ms']:.4f} ms/batch")
    print(f"  每token开销: {hardware_info['per_token_overhead_ms']:.6f} ms/token")
    
    # 调度决策示例
    print("\n⚡ 调度决策示例")
    print("-" * 30)
    scenarios = [
        (30, 512, "低延迟实时场景"),
        (50, 1024, "实时推理场景"),
        (100, 2048, "批处理场景"),
        (200, 4096, "高吞吐批处理")
    ]
    
    print(f"{'场景':15s} {'目标延迟':8s} {'Token预算':8s} {'推荐Batch':8s} {'预测延迟':8s} {'有效吞吐':10s}")
    print("-" * 75)
    
    for target_latency, token_budget, desc in scenarios:
        config = model.estimate_optimal_batch_config(target_latency, token_budget)
        
        if config:
            print(f"{desc:15s} {target_latency:8.0f} {token_budget:8d} "
                  f"{config['batch_size']:8.0f} {config['predicted_latency_ms']:8.1f} "
                  f"{config['effective_throughput']:10.4f}")
        else:
            print(f"{desc:15s} {target_latency:8.0f} {token_budget:8d} {'无解':8s} {'N/A':8s} {'N/A':10s}")
    
    return model, hardware_info, df


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("""
使用方法:
  python integration.py analyze [log_file_or_dir] [suffix] [--no-save]     # 使用原始吞吐饱和模型分析profiling数据
  python integration.py stable [log_file_or_dir] [suffix] [--no-save]      # 使用稳定集群调度模型分析
  python integration.py cluster [log_file_or_dir] [suffix] [--no-save]     # 集群调度专门分析（推荐用于异构集群）
  python integration.py compare [log_file_or_dir] [suffix] [--no-save]     # 比较不同模型性能
  python integration.py demo                                                # 运行演示
  
  参数说明:
  log_file_or_dir: profiling数据路径
  suffix: 可选的文件名后缀，用于区分不同的分析结果（如: a100, h100, v100等）
  --no-save: 可选参数，添加此参数将不保存模型和图片到文件系统
  
  示例:
  python integration.py analyze profiling_result a100              # 分析并保存所有输出
  python integration.py stable profiling_result h100 --no-save     # 分析但不保存文件
        """)
        return
    
    command = sys.argv[1]
    save_outputs = '--no-save' not in sys.argv
    
    # 解析参数
    log_path = sys.argv[2] if len(sys.argv) > 2 else 'profiling_result'
    suffix = sys.argv[3] if len(sys.argv) > 3 else 'a100'
    
    # 处理--no-save参数位置
    if suffix == '--no-save':
        suffix = 'a100'
    
    # 执行命令
    if command == 'analyze':
        model, df = analyze_with_modeling(log_path, suffix=suffix, save_outputs=save_outputs)
        save_model_if_needed(model, df, save_outputs, f'fitted_model_{suffix}.pkl')
    
    elif command == 'stable':
        model, df = analyze_with_modeling(log_path, use_stable_model=True, suffix=suffix, save_outputs=save_outputs)
        save_model_if_needed(model, df, save_outputs, f'stable_cluster_model_{suffix}.pkl')
    
    elif command == 'cluster':
        model, hardware_info, df = analyze_cluster_scheduling(log_path, suffix=suffix, save_outputs=save_outputs)
        save_model_if_needed(model, df, save_outputs, f'cluster_scheduling_model_{suffix}.pkl')
        if model and save_outputs:
            print_cluster_info(hardware_info)
    
    elif command == 'compare':
        df = read_profiling_data(log_path)
        if df is not None:
            compare_models(df, suffix, save_outputs)
    
    elif command == 'demo':
        from performance_model import demo_usage
        model, df = demo_usage()
        
    else:
        print(f"❌ 未知命令: {command}")
        print("💡 提示: 对于异构集群调度，推荐使用 'cluster' 命令")


def save_model_if_needed(model, df, save_outputs, filename):
    """如果需要保存，则保存模型"""
    if model and df is not None and save_outputs:
        model.save_model(f'./modeling/{filename}')
        print(f"\n💾 Model saved: {filename}")


def print_cluster_info(hardware_info):
    """打印集群调度信息"""
    print(f"\n🔑 关键集群调度参数:")
    print(f"   峰值吞吐量: {hardware_info['peak_throughput_tokens_per_ms']:.6f} tokens/ms")
    print(f"   硬件性能评分: {hardware_info['hardware_score']:.6f}")
    print(f"   批次效率系数: {hardware_info['batch_efficiency_factor']:.6f}")
    print(f"   Token效率系数: {hardware_info['token_efficiency_factor']:.6f}")


if __name__ == '__main__':
    main() 