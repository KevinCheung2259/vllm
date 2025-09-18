# Model Runtime vs Total Schedule Tokens 散点图生成器

该脚本用于生成模型运行时间与总调度token数量关系的散点图，支持多配置对比分析。

## 功能特性

- 🔄 **多配置对比**: 支持同时分析多个profiling文件夹，不同配置用不同颜色和标记显示
- 📊 **单配置分析**: 可以分析单个配置，支持按batch size着色
- 🎨 **美观图表**: 与其他paper_figs保持一致的样式和格式
- 📁 **灵活输入**: 支持目录或单个jsonl文件作为输入
- 💾 **高质量输出**: 300 DPI高清图片输出

## 数据字段说明

脚本读取的profiling数据应包含以下字段：
- `chunk_sizes`: 每个批次中各chunk的大小列表
- `model_run_duration_ms`: 模型运行时间（毫秒）
- `schedule_duration_ms`: 调度时间（毫秒，用于数据过滤）

## 使用方法

### 多配置对比模式

```bash
# 比较多个配置
python draw_runtime_vs_tokens.py \
    "Config A:/path/to/profiling_a" \
    "Config B:/path/to/profiling_b" \
    "Config C:/path/to/profiling_c" \
    --output multi_config_comparison.png \
    --title "Model Runtime vs Total Schedule Tokens"

# 使用目录名作为配置名
python draw_runtime_vs_tokens.py \
    /path/to/profiling_a \
    /path/to/profiling_b \
    /path/to/profiling_c
```

### 单配置分析模式

```bash
# 单配置，按batch size着色
python draw_runtime_vs_tokens.py \
    --single /path/to/profiling \
    --color-by-batch \
    --output single_config_analysis.png

# 单配置，单色显示
python draw_runtime_vs_tokens.py \
    --single /path/to/profiling \
    --output single_config_simple.png
```

### 参数说明

- `configs`: 多配置模式下的配置规格，格式为 `config_name:/path/to/profiling_dir`
- `--single`: 单配置模式，指定单个profiling目录路径
- `--output`, `-o`: 输出图片文件名（默认: model_runtime_vs_tokens.png）
- `--title`: 图表标题
- `--color-by-batch`: 在单配置模式下根据batch size着色
- `--verbose`, `-v`: 详细输出模式

## 数据处理逻辑

1. **Total Schedule Tokens计算**: 对每个batch，计算`chunk_sizes`列表的总和
2. **数据清洗**: 
   - 过滤掉调度时间超过300ms的数据点
   - 过滤掉模型运行时间超过200ms的数据点
3. **文件处理**: 去掉每个jsonl文件的前10行和后10行
4. **批次ID处理**: 在合并多个文件时，自动调整batch_id避免冲突

## 输出图表说明

### 多配置对比图
- X轴: Total Scheduled Tokens（总调度token数）
- Y轴: Model Run Time (ms)（模型运行时间）
- 不同配置用不同颜色和标记区分
- 图例显示在左上角

### 单配置分析图
- X轴: Total Scheduled Tokens
- Y轴: Model Run Time (ms)
- 可选择按batch size着色（使用viridis颜色映射）
- 包含颜色条说明batch size范围

## 示例使用场景

1. **性能对比**: 比较不同调度算法或参数设置的性能表现
2. **负载分析**: 观察模型运行时间随token数量的变化趋势
3. **批次大小影响**: 分析不同batch size对性能的影响
4. **系统调优**: 识别性能瓶颈和优化机会

## 注意事项

- 确保profiling数据格式正确，包含必要的字段
- 大型数据集可能需要较长的处理时间
- 建议使用SSD存储以提高文件读取性能
- 图表会自动设置合适的坐标轴范围（从0开始） 