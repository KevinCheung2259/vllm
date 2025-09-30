# Model Run Duration 分析文件夹

本文件夹包含了对 `total_scheduled_tokens=4096` 条件下 `model_run_duration_ms` 变化原因的完整分析。

## 📁 文件结构

### 📊 分析脚本
- `analyze_model_run_duration.py` - 基础分析脚本，筛选数据并进行初步分析
- `detailed_duration_analysis.py` - 深度分析脚本，包含特征重要性和详细模式分析

### 📈 可视化图表 (英文版)
- `model_run_duration_analysis.png` - 基础分析的12个子图可视化 (英文标题和标签)
- `detailed_duration_analysis.png` - 深度分析的9个子图可视化 (英文标题和标签)

### 📄 数据文件
- `filtered_data_4096.csv` - 筛选出的 total_scheduled_tokens=4096 的原始数据
- `enhanced_analysis_data.csv` - 增强分析数据，包含所有派生特征

### 📋 报告文档
- `ANALYSIS_REPORT.md` - 完整的分析报告，包含所有发现、结论和优化建议

## 🎯 主要发现

1. **Chunk大小标准差** 是影响运行时间的最重要因素 (相关性: 0.2251)
2. **并发竞争** 在超过12个请求后显著影响性能 (相关性: 0.1889)
3. **计算复杂度** 通过computed tokens数量影响性能 (相关性: 0.1731)
4. **内存访问模式** 的缓存缺失影响运行时间 (相关性: 0.1877)

## 🚀 快速开始

1. 查看完整分析报告：`cat ANALYSIS_REPORT.md`
2. 运行基础分析：`python analyze_model_run_duration.py`
3. 运行深度分析：`python detailed_duration_analysis.py`
4. 查看可视化结果：打开 `.png` 文件

## 📊 数据统计

- **分析记录数**: 3058 条
- **平均运行时间**: 104.28 ms
- **标准差**: 3.50 ms
- **变异系数**: 0.0335 (系统稳定性良好)
- **异常值比例**: 0.2%

## 📝 更新日志

### 2024年8月11日 - v2.0
- ✅ **重要更新**: 所有图表标题、坐标轴标签和图例已更新为英文
- ✅ 移除中文字体依赖，使用标准英文字体
- ✅ 优化图表文件保存路径
- ✅ 确保图表在国际化环境中的兼容性

---
*初始生成时间: 2024年8月11日*  
*英文化更新: 2024年8月11日* 