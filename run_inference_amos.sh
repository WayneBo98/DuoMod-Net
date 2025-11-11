#!/bin/bash
# 这是一个统一的推理和评估脚本（支持多 seed 并自动汇总统计）

# --- 1. 用户配置区 (请根据您的实际情况修改这里的变量) ---

# 实验根目录模板，{seed} 会被替换
# 🎯 公共日志根目录（所有实验共享）
LOG_ROOT="/data/wangbo/CissMOS/logs/amos2p/uamt/base"

# 实验根目录模板，{seed} 会被替换
EXP_ROOT_TEMPLATE="${LOG_ROOT}/seed_{seed}"

# 汇总输出文件
AGGREGATE_OUTPUT_FILE="${LOG_ROOT}/multi_seed_aggregated_results.txt"
# 输入数据路径
NPY_DATA_PATH="/data/wangbo/CissMOS/Datasets/AMOS22_1.5_2.0_npy/imagesVa"
ORIGINAL_NII_PATH="/data/wangbo/CissMOS/Datasets/Amos22/imagesVa"
GT_PATH="/data/wangbo/CissMOS/Datasets/Amos22/labelsVa"

# 其他参数
GPU_ID="3"
TASK_NAME="amos"
OVERLAP=0.5
NUM_CLASSES=16

# 要测试的 seeds
SEEDS=(0 1 2)

# --- 2. 自动派生路径 & 初始化 ---

set -e

echo "🧩 Starting multi-seed evaluation workflow..."

# --- 3. 循环处理每个 seed ---

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "================================================="
    echo "🌱 Processing SEED: $seed"
    echo "================================================="

    # 动态替换路径中的 {seed}
    EXP_ROOT=$(echo "$EXP_ROOT_TEMPLATE" | sed "s/{seed}/$seed/g")

    MODEL_PATH="${EXP_ROOT}/ckpts/best_model.pth"
    PRED_PATH="${EXP_ROOT}/predictions"
    OUTPUT_FILE="${EXP_ROOT}/evaluation_results.txt"

    echo "📁 Model Path: ${MODEL_PATH}"
    echo "📁 Prediction Output Path: ${PRED_PATH}"
    echo "📁 Evaluation Result File: ${OUTPUT_FILE}"

    # --- 3.1 执行推理 ---
    echo "🚀 Step 1: Running Inference (test_cps.py)"
    python test_cps.py \
        --npy_path "${NPY_DATA_PATH}" \
        --original_nii_path "${ORIGINAL_NII_PATH}" \
        --output_path "${PRED_PATH}" \
        --model_path "${MODEL_PATH}" \
        --gpu "${GPU_ID}" \
        --exp "uamt" \
        --task "${TASK_NAME}" \
        --overlap "${OVERLAP}"
done