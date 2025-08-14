#!/bin/bash
# SLA感知调度器配置示例
# 使用方法: source config_examples.sh <scenario>

print_usage() {
    echo "SLA感知调度器配置示例"
    echo "使用方法: source $0 <scenario>"
    echo ""
    echo "可用场景:"
    echo "  development  - 开发环境配置 (详细日志, 保守参数)"
    echo "  testing      - 测试环境配置 (性能监控, 中等参数)"
    echo "  production   - 生产环境配置 (优化性能, 严格SLA)"
    echo "  disabled     - 禁用SLA调度器"
    echo "  debug        - 调试配置 (所有日志开启)"
}

set_development_config() {
    echo "🔧 设置开发环境配置..."
    
    # 功能开关
    export VLLM_SLA_SCHEDULER_ENABLED=true
    export VLLM_SLA_FALLBACK_ON_ERROR=true
    
    # SLA参数 (宽松)
    export VLLM_SLO_TPOT_MS=100.0
    export VLLM_SLO_TTFT_MS=1000.0
    export VLLM_SLA_MIN_BATCH_TIME_MS=20.0
    export VLLM_SLA_QUEUE_THRESHOLD=10
    
    # 性能模型参数 (频繁更新)
    export VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.20
    export VLLM_SLA_MIN_SAMPLES=20
    export VLLM_SLA_BUFFER_SIZE=500
    
    # 优化算法参数 (保守)
    export VLLM_SLA_MAX_BATCH_SEARCH=16
    export VLLM_SLA_OPT_TIMEOUT_MS=2.0
    
    # 调试选项
    export VLLM_SLA_VERBOSE=true
    export VLLM_SLA_PERF_LOG=true
    
    echo "✅ 开发环境配置已设置"
    echo "   - 宽松SLA: TPOT=${VLLM_SLO_TPOT_MS}ms"
    echo "   - 详细日志: 启用"
    echo "   - 自动回退: 启用"
}

set_testing_config() {
    echo "🧪 设置测试环境配置..."
    
    # 功能开关
    export VLLM_SLA_SCHEDULER_ENABLED=true
    export VLLM_SLA_FALLBACK_ON_ERROR=true
    
    # SLA参数 (中等)
    export VLLM_SLO_TPOT_MS=75.0
    export VLLM_SLO_TTFT_MS=750.0
    export VLLM_SLA_MIN_BATCH_TIME_MS=18.0
    export VLLM_SLA_QUEUE_THRESHOLD=7
    
    # 性能模型参数 (平衡)
    export VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.15
    export VLLM_SLA_MIN_SAMPLES=50
    export VLLM_SLA_BUFFER_SIZE=800
    
    # 优化算法参数 (标准)
    export VLLM_SLA_MAX_BATCH_SEARCH=24
    export VLLM_SLA_OPT_TIMEOUT_MS=1.5
    
    # 调试选项
    export VLLM_SLA_VERBOSE=false
    export VLLM_SLA_PERF_LOG=true
    
    echo "✅ 测试环境配置已设置"
    echo "   - 中等SLA: TPOT=${VLLM_SLO_TPOT_MS}ms"
    echo "   - 性能监控: 启用"
    echo "   - 自动回退: 启用"
}

set_production_config() {
    echo "🚀 设置生产环境配置..."
    
    # 功能开关
    export VLLM_SLA_SCHEDULER_ENABLED=true
    export VLLM_SLA_FALLBACK_ON_ERROR=false  # 生产环境不回退
    
    # SLA参数 (严格)
    export VLLM_SLO_TPOT_MS=50.0
    export VLLM_SLO_TTFT_MS=500.0
    export VLLM_SLA_MIN_BATCH_TIME_MS=15.0
    export VLLM_SLA_QUEUE_THRESHOLD=5
    
    # 性能模型参数 (稳定)
    export VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.10
    export VLLM_SLA_MIN_SAMPLES=100
    export VLLM_SLA_BUFFER_SIZE=1000
    export VLLM_SLA_MODEL_CONFIDENCE=0.85
    
    # 优化算法参数 (激进)
    export VLLM_SLA_MAX_BATCH_SEARCH=32
    export VLLM_SLA_OPT_TIMEOUT_MS=1.0
    
    # 调试选项 (最小)
    export VLLM_SLA_VERBOSE=false
    export VLLM_SLA_PERF_LOG=true
    
    echo "✅ 生产环境配置已设置"
    echo "   - 严格SLA: TPOT=${VLLM_SLO_TPOT_MS}ms"
    echo "   - 高性能优化: 启用"
    echo "   - 自动回退: 禁用"
    echo "   ⚠️  请确保已充分测试！"
}

set_disabled_config() {
    echo "❌ 禁用SLA调度器..."
    
    export VLLM_SLA_SCHEDULER_ENABLED=false
    
    # 清除其他SLA相关配置
    unset VLLM_SLO_TPOT_MS
    unset VLLM_SLO_TTFT_MS
    unset VLLM_SLA_MIN_BATCH_TIME_MS
    unset VLLM_SLA_QUEUE_THRESHOLD
    unset VLLM_SLA_MODEL_UPDATE_THRESHOLD
    unset VLLM_SLA_MIN_SAMPLES
    unset VLLM_SLA_VERBOSE
    
    echo "✅ SLA调度器已禁用，将使用原有调度逻辑"
}

set_debug_config() {
    echo "🐛 设置调试配置..."
    
    # 功能开关
    export VLLM_SLA_SCHEDULER_ENABLED=true
    export VLLM_SLA_FALLBACK_ON_ERROR=true
    
    # SLA参数 (宽松，便于调试)
    export VLLM_SLO_TPOT_MS=200.0
    export VLLM_SLO_TTFT_MS=2000.0
    export VLLM_SLA_MIN_BATCH_TIME_MS=30.0
    export VLLM_SLA_QUEUE_THRESHOLD=15
    
    # 性能模型参数 (频繁更新，小样本)
    export VLLM_SLA_MODEL_UPDATE_THRESHOLD=0.30
    export VLLM_SLA_MIN_SAMPLES=10
    export VLLM_SLA_BUFFER_SIZE=200
    
    # 优化算法参数 (保守，长超时)
    export VLLM_SLA_MAX_BATCH_SEARCH=8
    export VLLM_SLA_OPT_TIMEOUT_MS=5.0
    
    # 调试选项 (全部启用)
    export VLLM_SLA_VERBOSE=true
    export VLLM_SLA_PERF_LOG=true
    
    # 额外的调试环境变量
    export VLLM_SCHEDULER_PROFILING_CONSOLE=true
    
    echo "✅ 调试配置已设置"
    echo "   - 超宽松SLA: TPOT=${VLLM_SLO_TPOT_MS}ms"
    echo "   - 全部日志: 启用"
    echo "   - 小样本快速更新: 启用"
    echo "   - Console输出: 启用"
}

print_current_config() {
    echo ""
    echo "📋 当前SLA调度器配置:"
    echo "================================"
    echo "功能开关:"
    echo "  VLLM_SLA_SCHEDULER_ENABLED=${VLLM_SLA_SCHEDULER_ENABLED:-未设置}"
    echo "  VLLM_SLA_FALLBACK_ON_ERROR=${VLLM_SLA_FALLBACK_ON_ERROR:-未设置}"
    echo ""
    echo "SLA参数:"
    echo "  VLLM_SLO_TPOT_MS=${VLLM_SLO_TPOT_MS:-未设置}"
    echo "  VLLM_SLO_TTFT_MS=${VLLM_SLO_TTFT_MS:-未设置}"
    echo "  VLLM_SLA_MIN_BATCH_TIME_MS=${VLLM_SLA_MIN_BATCH_TIME_MS:-未设置}"
    echo "  VLLM_SLA_QUEUE_THRESHOLD=${VLLM_SLA_QUEUE_THRESHOLD:-未设置}"
    echo ""
    echo "性能模型:"
    echo "  VLLM_SLA_MODEL_UPDATE_THRESHOLD=${VLLM_SLA_MODEL_UPDATE_THRESHOLD:-未设置}"
    echo "  VLLM_SLA_MIN_SAMPLES=${VLLM_SLA_MIN_SAMPLES:-未设置}"
    echo ""
    echo "调试选项:"
    echo "  VLLM_SLA_VERBOSE=${VLLM_SLA_VERBOSE:-未设置}"
    echo "  VLLM_SLA_PERF_LOG=${VLLM_SLA_PERF_LOG:-未设置}"
    echo "================================"
}

# 主逻辑
if [ $# -eq 0 ]; then
    print_usage
    return 1 2>/dev/null || exit 1
fi

case "$1" in
    "development"|"dev")
        set_development_config
        ;;
    "testing"|"test")
        set_testing_config
        ;;
    "production"|"prod")
        set_production_config
        ;;
    "disabled"|"off")
        set_disabled_config
        ;;
    "debug")
        set_debug_config
        ;;
    "show"|"current")
        print_current_config
        ;;
    *)
        echo "❌ 未知场景: $1"
        print_usage
        return 1 2>/dev/null || exit 1
        ;;
esac

# 显示当前配置
if [ "$1" != "show" ] && [ "$1" != "current" ]; then
    print_current_config
fi

echo ""
echo "💡 提示:"
echo "   - 使用 'source $0 show' 查看当前配置"
echo "   - 配置生效需要重启vLLM服务"
echo "   - 可以随时切换配置场景"
