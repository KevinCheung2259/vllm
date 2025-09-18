#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例脚本：演示如何使用 draw_runtime_vs_tokens.py 生成散点图
"""

import os
import sys
from pathlib import Path
from draw_runtime_vs_tokens import ModelRuntimeVsTokensPlotGenerator

def example_single_config():
    """示例：单配置分析"""
    print("🔍 示例：单配置分析")
    
    # 假设您有一个profiling数据目录
    profiling_dir = "../../exp/profiling_result_h100"  # 示例路径，请根据实际情况修改
    
    if not Path(profiling_dir).exists():
        print(f"❌ 示例目录 {profiling_dir} 不存在")
        print("请修改 profiling_dir 为实际的profiling数据目录")
        return
    
    # 创建绘图器
    plotter = ModelRuntimeVsTokensPlotGenerator(verbose=True)
    
    # 读取数据
    df = plotter.read_profiling_data(profiling_dir, "H100 Config")
    
    if df is None or df.empty:
        print("❌ 无法读取数据")
        return
    
    # 生成图表（按batch size着色）
    fig = plotter.generate_single_config_plot(
        df, 
        config_name="H100 Config",
        save_path="single_config_example.png",
        title="Model Runtime vs Total Tokens (Single Config)",
        color_by_batch_size=True
    )
    
    print("✅ 单配置图表生成完成")

def example_multi_config():
    """示例：多配置对比"""
    print("🔍 示例：多配置对比")
    
    # 示例配置目录列表
    config_dirs = {
        "H100": "../../exp/profiling_result_h100",
        "A100": "../../exp/profiling_result_a100", 
        "A6000": "../../exp/profiling_result_a6000"
    }
    
    # 检查哪些目录存在
    valid_configs = {}
    for name, path in config_dirs.items():
        if Path(path).exists():
            valid_configs[name] = path
        else:
            print(f"⚠️  目录 {path} 不存在，跳过配置 {name}")
    
    if len(valid_configs) < 2:
        print("❌ 需要至少2个有效的配置目录进行对比")
        print("请检查并修改 config_dirs 中的路径")
        return
    
    # 创建绘图器
    plotter = ModelRuntimeVsTokensPlotGenerator(verbose=True)
    
    # 读取所有配置的数据
    config_data_dict = {}
    for config_name, config_path in valid_configs.items():
        df = plotter.read_profiling_data(config_path, config_name)
        if df is not None and not df.empty:
            config_data_dict[config_name] = df
    
    if not config_data_dict:
        print("❌ 没有成功读取任何配置数据")
        return
    
    # 生成多配置对比图
    fig = plotter.generate_multi_config_plot(
        config_data_dict,
        save_path="multi_config_example.png",
        title="Model Runtime vs Total Tokens (Multi-Config Comparison)"
    )
    
    print("✅ 多配置对比图表生成完成")

def main():
    """主函数"""
    print("📊 Model Runtime vs Total Schedule Tokens 绘图工具示例")
    print("=" * 60)
    
    # 确保当前目录正确
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # 运行单配置示例
        example_single_config()
        print()
        
        # 运行多配置示例  
        example_multi_config()
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        return
    
    print()
    print("🎉 示例运行完成！")
    print("📁 生成的图表文件:")
    
    output_files = ["single_config_example.png", "multi_config_example.png"]
    for file in output_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (未生成)")

if __name__ == '__main__':
    main() 