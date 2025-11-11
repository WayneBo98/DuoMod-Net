#!/bin/bash
# run_evaluation_pipeline.sh
# 这是一个统一的、支持多方法的评估与聚合脚本。
#
# 工作流程:
# 1. 对每个方法、每个 seed:
#    - 调用 eval_metrics.py 计算详细指标并保存为 per-sample CSV。
# 2. 对每个方法:
#    - 调用 aggregate_and_prepare_stats.py 汇总其所有 seed 的结果，
#      生成最终的 Mean±Std 总结，并为 p-value 计算准备聚合后的 CSV。
#
# 使用前请确保:
# - eval_metrics.py 和 aggregate_and_prepare_stats.py 在当前目录或 PATH 中。
# - Python 环境已激活，且已安装 pandas, numpy, openpyxl, scipy, simpleitk。

# --- 1. 用户配置区 (✅ 请根据您的实际情况修改这里的变量) ---

# 数据集和通用设置
GT_PATH="/data/wangbo/CissMOS/Datasets/Amos22/labelsVa"
NUM_CLASSES=16 # 类别总数 (含背景)
SEEDS=(0 1 2)    # 要处理的 seed 列表

# --- 方法配置 ---
# 在这里定义所有需要评估的方法。
# 格式: "方法名称;日志根目录"
# - 方法名称: 用于命名输出文件，例如 "cps", "dhc", "ours"
# - 日志根目录: 该方法包含所有 seed 子目录的基础路径
METHODS_TO_PROCESS=(
    "cps;/data/wangbo/CissMOS/logs/amos10p/cps/test"
    "uamt;/data/wangbo/CissMOS/logs/amos10p/uamt/base"
    "js_divergence+dar;/data/wangbo/CissMOS/logs/amos10p/afr_modification/js_divergence"
    # "js_divergence;/data/wangbo/CissMOS/logs/amos/afr_modification/js_divergence_ce"
    # "dar;/data/wangbo/CissMOS/logs/amos/distribution_aware_reweighting/wdiceloss_newmean"
    "dst;/data/wangbo/CissMOS/logs/amos10p/dst/test"
    "dmd;/data/wangbo/CissMOS/logs/amos10p/dmd/test"
    "slcnet;/data/wangbo/CissMOS/logs/amos10p/slcnet/test"
    "dycon;/data/wangbo/CissMOS/logs/amos10p/dycon/test"
    # "dhc;/data/wangbo/CissMOS/logs/amos2p/dhc/new"
    # 添加更多方法...
    # "ours;/data/wangbo/CissMOS/logs/amos2p/ours/final_run"
)

# --- Python 脚本名称配置 ---
EVAL_SCRIPT="eval.py"
AGGREGATE_SCRIPT="aggregate.py"

# --- 2. 脚本主逻辑 (通常无需修改) ---

set -e # 如果任何命令失败，则立即退出脚本

echo "🚀🚀🚀 Starting Full Evaluation & Aggregation Pipeline 🚀🚀🚀"

# 存储所有方法聚合后 CSV 文件的路径，为 p-value 计算做准备
AGGREGATED_CSV_PATHS=()

# --- 循环处理每个方法 ---
for method_info in "${METHODS_TO_PROCESS[@]}"; do
    # 解析方法名称和路径
    IFS=';' read -r METHOD_NAME LOG_ROOT <<< "$method_info"

    echo ""
    echo "================================================="
    echo "Processing Method: ${METHOD_NAME}"
    echo "Log Root: ${LOG_ROOT}"
    echo "================================================="

    # --- 3. 为该方法的每个 seed 运行评估 ---
    echo "--- Step 1: Evaluating each seed for '${METHOD_NAME}' ---"
    for seed in "${SEEDS[@]}"; do
        echo "  🌱 Evaluating seed: $seed..."

        PRED_PATH="${LOG_ROOT}/seed_${seed}/predictions"
        OUTPUT_CSV="${LOG_ROOT}/seed_${seed}/result.csv" # 统一使用 .csv

        # 检查预测目录是否存在
        if [ ! -d "$PRED_PATH" ]; then
            echo "  ⚠️ Warning: Prediction path not found, skipping: ${PRED_PATH}"
            continue
        fi

        python "${EVAL_SCRIPT}" \
            --pred_path "${PRED_PATH}" \
            --gt_path "${GT_PATH}" \
            --num_classes "${NUM_CLASSES}" \
            --output_csv "${OUTPUT_CSV}"
        
        echo "  ✅ Evaluation for seed $seed finished. Per-sample results saved to: ${OUTPUT_CSV}"
    done

    # --- 4. 汇总该方法的所有 seed 结果 ---
    echo "--- Step 2: Aggregating results for '${METHOD_NAME}' ---"

    AGGREGATED_CSV_OUTPUT="${LOG_ROOT}/${METHOD_NAME}_aggregated_metrics.csv"
    SUMMARY_OUTPUT="${LOG_ROOT}/${METHOD_NAME}_summary_results.txt"

    python "${AGGREGATE_SCRIPT}" \
        --method_base_dir "${LOG_ROOT}" \
        --results_filename "result" \
        --aggregated_csv_output "${AGGREGATED_CSV_OUTPUT}" \
        --summary_output "${SUMMARY_OUTPUT}"

    echo "✅ Aggregation for '${METHOD_NAME}' finished."
    echo "   - Final Mean±Std summary: ${SUMMARY_OUTPUT}"
    echo "   - Aggregated CSV for p-value test: ${AGGREGATED_CSV_OUTPUT}"

    # 将聚合后的 CSV 路径添加到列表中
    AGGREGATED_CSV_PATHS+=("${AGGREGATED_CSV_OUTPUT}")
done


# --- 5. 结束与后续步骤提示 ---
echo ""
echo "================================================="
echo "🎉🎉🎉 ALL METHODS PROCESSED SUCCESSFULLY! 🎉🎉🎉"
echo "================================================="
echo "You can now find the final Mean±Std summary for each method in its log directory."
echo ""
echo "下一步: P-value Calculation"
echo "---------------------------------"
echo "The following aggregated CSV files have been generated and are ready for statistical testing:"
for path in "${AGGREGATED_CSV_PATHS[@]}"; do
    echo "  - ${path}"
done
echo ""
echo "You can now run your p-value calculation script using these files as input."
echo "Example command for a hypothetical p-value script:"
echo "python calculate_p_values.py --ours_csv [path_to_ours_aggregated.csv] --competitor_csvs [paths_to_other_csvs]"