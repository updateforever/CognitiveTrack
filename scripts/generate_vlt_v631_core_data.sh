#!/bin/bash
# VLT-v6.3.1 Core SFT 数据生成启动脚本

set -e

# ============================================================================
# 配置参数
# ============================================================================

# vLLM API 配置（确保 Qwen2.5-VL-32B 已部署在 8000 端口）
export LOCAL_VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_VLLM_API_KEY="local-test-key"

# 输出目录
OUTPUT_DIR="/data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft"

# Python 环境
PYTHON_BIN="python"

# ============================================================================
# 检查 vLLM 服务
# ============================================================================

echo "=========================================="
echo "VLT-v6.3.1 Core SFT Data Generation"
echo "=========================================="
echo ""

echo "Checking vLLM service at ${LOCAL_VLLM_BASE_URL}..."
if curl -s "${LOCAL_VLLM_BASE_URL}/models" > /dev/null 2>&1; then
    echo "✅ vLLM service is running"
else
    echo "❌ vLLM service is NOT running at ${LOCAL_VLLM_BASE_URL}"
    echo "Please start vLLM first:"
    echo "  bash scripts/start_vllm_qwen25_vl_32b.sh"
    exit 1
fi
echo ""

# ============================================================================
# 生成训练数据
# ============================================================================

echo "Starting data generation..."
echo "Output directory: ${OUTPUT_DIR}"
echo ""

${PYTHON_BIN} tracking/synthesize_vlt_v631_core_data.py

echo ""
echo "=========================================="
echo "✅ Data generation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Visualize samples:"
echo "   python scripts/visualize_training_samples.py --num_samples 50"
echo ""
echo "2. Check generation stats:"
echo "   cat ${OUTPUT_DIR}/generation_stats.json"
echo ""
