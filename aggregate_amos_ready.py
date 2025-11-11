#!/usr/bin/env python3
# aggregate_and_prepare_stats.py - 自动查找并汇总多个 seed 结果，计算 Mean±Std，
#                                并为 p-value 计算准备样本级聚合数据

import os
import re # 用于解析文件内容 (如果需要，目前是直接用 pandas 读取)
import numpy as np
import pandas as pd
import argparse
import glob # 用于文件查找
from tqdm import tqdm

# --- 配置区 ---
# AMOS 类别名 (请根据您的数据集和 eval_metrics.py 脚本调整)
# 示例：AMOS
CLASS_NAMES = [
    'spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus',
    'liver', 'stomach', 'aorta', 'inferior vena cava', 'pancreas',
    'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate/uterus'
]
# 示例：WORD (如果您的 eval_metrics.py 输出了这些类别名对应的列)
# CLASS_NAMES = [
#     'liver', 'spleen', 'kidney L', 'kidney R', 'stomach', 'gallbladder', 'esophagus',
#     'pancreas', 'duodenum', 'colon', 'intestine', 'adrenal', 'rectum', 'bladder',
#     'femur L', 'femur R'
# ]

NUM_FOREGROUND_CLASSES = len(CLASS_NAMES)

# 定义 Tail Classes (确保名称与 CLASS_NAMES 匹配)
# 示例：AMOS
TAIL_CLASSES = [
    'esophagus',
    'gallbladder',
    'duodenum',
    'right adrenal gland',
    'left adrenal gland'
]
# 示例：WORD (根据实际情况定义)
# TAIL_CLASSES = [
#     'gallbladder', 'esophagus', 'duodenum', 'adrenal', 'rectum'
# ]


# --- 辅助函数 ---
def get_metric_columns(metric_prefix, class_indices):
    """根据指标前缀和类别索引生成列名列表"""
    # class_indices 是 0-based, 但列名是 1-based (e.g., Dice_Class1)
    return [f"{metric_prefix}_Class{idx + 1}" for idx in class_indices]

# --- 主函数 ---
def main(args):
    # --- 1. 自动查找 seed 结果文件 ---
    seed_dirs = sorted(glob.glob(os.path.join(args.method_base_dir, args.seed_pattern)))
    result_files_found = []
    for seed_dir in seed_dirs:
        # 兼容 .csv 和 .xlsx
        potential_csv = os.path.join(seed_dir, args.results_filename + ".csv")
        potential_xlsx = os.path.join(seed_dir, args.results_filename + ".xlsx")

        if os.path.exists(potential_csv):
            result_files_found.append(potential_csv)
        elif os.path.exists(potential_xlsx):
            result_files_found.append(potential_xlsx)
        else:
            print(f"⚠️ Warning: Results file '{args.results_filename}.(csv/xlsx)' not found in '{seed_dir}'. Skipping this seed directory.")

    if len(result_files_found) < args.min_seeds:
        print(f"❌ Error: Need at least {args.min_seeds} valid result files, found {len(result_files_found)} matching pattern '{args.seed_pattern}/{args.results_filename}.(csv/xlsx)' in '{args.method_base_dir}'.")
        return

    print(f"🔍 Found {len(result_files_found)} result files to aggregate:")
    for fpath in result_files_found:
        print(f"  - {fpath}")

    # --- 2. 读取找到的文件 ---
    dfs = []
    valid_seeds_read = 0
    for file_path in result_files_found:
        try:
            if file_path.endswith(".csv"):
                # 尝试自动检测分隔符，以防万一不是逗号
                df = pd.read_csv(file_path, engine='python', sep=None, on_bad_lines='warn')
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                print(f"Unsupported file type: {file_path}")
                continue

            # 替换无限值为 NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            if "Filename" not in df.columns or df["Filename"].duplicated().any():
                print(f"⚠️ Warning: File '{file_path}' lacks unique 'Filename' column. Skipping.")
                continue
            df.set_index("Filename", inplace=True)
            dfs.append(df)
            valid_seeds_read += 1
        except Exception as e:
            print(f"❌ Error reading '{file_path}': {e}")

    # 再次检查读取成功的文件数
    if valid_seeds_read < args.min_seeds:
        print(f"❌ Error: Successfully read only {valid_seeds_read} files, need at least {args.min_seeds}.")
        return

    # --- 3. 检查样本是否一致 (基于索引 'Filename') ---
    base_index = dfs[0].index
    for i in range(1, len(dfs)):
        if not dfs[i].index.equals(base_index):
            print("❌ Error: Sample filenames (index) mismatch between result files.")
            # 尝试找到共同的样本进行分析
            common_index = base_index.intersection(dfs[i].index)
            if len(common_index) == 0:
                print("❌ Error: No common samples found between files. Cannot proceed.")
                return
            print(f"⚠️ Warning: Found only {len(common_index)} common samples. Proceeding with intersection.")
            base_index = common_index
            # 筛选所有 DataFrame 以包含共同样本
            dfs = [df.loc[base_index] for df in dfs]

    print(f"✅ Processing {len(dfs)} result files with {len(base_index)} common samples each.")

    # --- 4. 准备列名 ---
    fg_indices = list(range(NUM_FOREGROUND_CLASSES))
    try:
        # 确保 CLASS_NAMES 是最新的，并且与 CSV 文件中的 Class 列对应
        # 注意：这里假设 CLASS_NAMES 的顺序与 CSV 中 Class1, Class2... 的顺序一致
        tail_indices = [CLASS_NAMES.index(name) for name in TAIL_CLASSES if name in CLASS_NAMES]
        if len(tail_indices) != len(TAIL_CLASSES):
             missing = set(TAIL_CLASSES) - set(CLASS_NAMES)
             print(f"⚠️ Warning: Some defined tail classes not found in CLASS_NAMES: {missing}")
    except ValueError as e:
        print(f"❌ Error: Tail class name '{str(e).split()[0]}' not found in CLASS_NAMES.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing class names: {e}")
        return


    cols_fg_dice = get_metric_columns("Dice", fg_indices)
    cols_fg_asd = get_metric_columns("ASD", fg_indices)

    cols_tail_dice = get_metric_columns("Dice", tail_indices)
    cols_tail_asd = get_metric_columns("ASD", tail_indices)

    # --- 5. 计算每个样本跨 Seed 的平均指标 (用于 p-value 计算) ---
    aggregated_data = pd.DataFrame(index=base_index) # 创建新的 DataFrame 存储聚合结果

    metrics_to_aggregate = {
        "Avg_FG_Dice": cols_fg_dice,
        "Avg_FG_ASD": cols_fg_asd,
        # 只有在定义了 tail_indices 时才计算尾部指标
        **({"Avg_Tail_Dice": cols_tail_dice} if tail_indices else {}),
        **({"Avg_Tail_ASD": cols_tail_asd} if tail_indices else {}),
    }

    print("⏳ Calculating per-sample averages across seeds...")
    for agg_metric_name, class_cols in tqdm(metrics_to_aggregate.items(), desc="Aggregating metrics"):
        # 确保只使用在所有 DataFrame 中都存在的列
        valid_cols_list = []
        for df in dfs:
            existing_cols = [col for col in class_cols if col in df.columns]
            if not existing_cols: # 如果一个df连一列都没有，则无法计算
                print(f"⚠️ Warning: No columns found for metric {agg_metric_name} in one of the seed files. Skipping this metric for per-sample aggregation.")
                aggregated_data[agg_metric_name] = np.nan
                break # 跳出内层循环，处理下一个聚合指标
            valid_cols_list.append(existing_cols)
        else: # 如果 for 循环正常结束 (没有 break)
            # 使用存在的列进行提取和堆叠
            try:
                # 重新索引以确保样本顺序一致，然后提取有效列的值
                seed_metric_values = [df.reindex(base_index)[vcc].values for df, vcc in zip(dfs, valid_cols_list)]

                # 堆叠成 3D NumPy 数组: (n_samples, n_classes_for_metric_in_this_seed, n_seeds)
                # 注意：不同 seed 的 n_classes 可能不同（如果某列完全缺失）
                # 我们需要先计算每个 seed 的样本内平均值，再跨 seed 平均
                sample_means_per_seed = []
                for seed_values in seed_metric_values:
                    # 对每个样本计算跨类别的平均值 (axis=1)
                    sample_means_per_seed.append(np.nanmean(seed_values, axis=1))

                # 现在 sample_means_per_seed 是一个列表，每个元素是 shape (n_samples,) 的数组
                # 将它们堆叠起来计算最终的跨 seed 平均值
                stacked_sample_means = np.stack(sample_means_per_seed, axis=-1) # shape: (n_samples, n_seeds)
                final_mean_per_sample = np.nanmean(stacked_sample_means, axis=1) # shape: (n_samples,)

                aggregated_data[agg_metric_name] = final_mean_per_sample

            except Exception as e:
                 print(f"❌ Error during stacking/averaging for {agg_metric_name}: {e}. Skipping.")
                 aggregated_data[agg_metric_name] = np.nan


    # --- 6. 保存聚合后的样本级数据到 CSV (用于 p-value) ---
    try:
        # 确保存储路径的目录存在
        os.makedirs(os.path.dirname(args.aggregated_csv_output), exist_ok=True)
        aggregated_data.to_csv(args.aggregated_csv_output, float_format='%.6f', na_rep='NaN')
        print(f"\n✅ Per-sample aggregated metrics saved to: {args.aggregated_csv_output}")
    except Exception as e:
        print(f"\n❌ Error saving aggregated CSV to '{args.aggregated_csv_output}': {e}")

    # --- 7. 计算最终的 Mean ± Std (用于论文表格) ---
    print("\n📊 Calculating final Mean ± Std across seeds...")

    final_results = {} # 存储最终的 Mean±Std 字符串

    # a) 计算 Per-Class 指标的 Mean±Std (跨 Seeds)
    per_class_means_across_seeds = {} # e.g., {"Dice": [seed0_avg_cls1, seed1_avg_cls1, ...]}
    per_class_stds_across_seeds = {}
    
    for metric_prefix in ["Dice", "ASD"]:
        means_for_metric = {} # { "Dice_Class1": [seed0_avg, seed1_avg, seed2_avg], ... }
        for i in range(NUM_FOREGROUND_CLASSES):
            col_name = f"{metric_prefix}_Class{i+1}"
            seed_averages = []
            for df in dfs:
                if col_name in df.columns:
                    # 计算该 seed 在该类别上的平均值 (跨所有样本)
                    seed_class_mean = np.nanmean(df[col_name])
                    seed_averages.append(seed_class_mean)
                else:
                    seed_averages.append(np.nan) # 如果某 seed 缺失该列

            # 过滤掉 NaN 后再计算
            valid_seed_averages = [avg for avg in seed_averages if not np.isnan(avg)]
            if len(valid_seed_averages) >= args.min_seeds: # 确保有足够数据点计算 Std
                 # 计算跨 seed 的均值和标准差
                 per_class_means_across_seeds.setdefault(metric_prefix, {})[col_name] = np.nanmean(valid_seed_averages)
                 per_class_stds_across_seeds.setdefault(metric_prefix, {})[col_name] = np.nanstd(valid_seed_averages)
            else:
                 # 数据点不足，标记为 NaN
                 per_class_means_across_seeds.setdefault(metric_prefix, {})[col_name] = np.nan
                 per_class_stds_across_seeds.setdefault(metric_prefix, {})[col_name] = np.nan


    # b) 计算 Overall 和 Tail Avg 指标的最终 Mean±Std (跨 Seeds)
    #    我们需要先计算出每个 seed 的 Overall/Tail 平均值
    seed_level_aggregates = {agg_metric: [] for agg_metric in aggregated_data.columns}

    for df in dfs: # 遍历每个 seed 的 DataFrame
        # 确保 DataFrame 索引与 base_index 一致 (如果之前做了交集处理)
        df_reindexed = df.reindex(base_index)
        for agg_metric_name, class_cols in metrics_to_aggregate.items():
            valid_cols = [col for col in class_cols if col in df_reindexed.columns]
            if not valid_cols:
                seed_level_aggregates[agg_metric_name].append(np.nan)
                continue
            # 计算该 seed 在所有样本上，跨指定类别的平均值
            # 1. 先计算每个样本跨类别的平均值
            sample_means = np.nanmean(df_reindexed[valid_cols].values, axis=1)
            # 2. 再计算所有样本的平均值
            seed_overall_mean = np.nanmean(sample_means)
            seed_level_aggregates[agg_metric_name].append(seed_overall_mean)

    # 现在 seed_level_aggregates 包含了每个聚合指标的 seed 级别平均值列表
    for agg_metric_name, seed_means in seed_level_aggregates.items():
        valid_seed_means = [m for m in seed_means if not np.isnan(m)]
        if len(valid_seed_means) >= args.min_seeds:
            final_mean = np.nanmean(valid_seed_means)
            final_std = np.nanstd(valid_seed_means) # <--- 正确的标准差！

            # 格式化输出
            if "Dice" in agg_metric_name:
                 final_results[agg_metric_name] = f"{final_mean*100:.2f} ± {final_std*100:.2f}"
            else: # ASD 保留 3 位小数
                 final_results[agg_metric_name] = f"{final_mean:.3f} ± {final_std:.3f}"
        else:
            final_results[agg_metric_name] = 'N/A' # 数据点不足

    # --- 8. 输出最终 Mean ± Std 结果到文件 ---
    try:
        # 确保存储路径的目录存在
        os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
        with open(args.summary_output, 'w') as f:
            f.write(f"📊 Aggregated Summary Results (Mean ± Std from {len(dfs)} seeds)\n")
            f.write(f"Method Base Directory: {args.method_base_dir}\n")
            f.write("="*70 + "\n\n")

            # --- Per-Class Metrics (使用新的计算结果) ---
            f.write("🔷 Per-Class Metrics:\n")
            f.write("-" * 50 + "\n")
            
            # --- 修改了表头和格式 ---
            header_format = f"{{:<{args.class_name_width}}} | {{:<18}} | {{:<18}}"
            row_format = f"{{:<{args.class_name_width}}} | {{:>18}} | {{:>18}}"
            f.write(header_format.format('Class', 'Dice (%)', 'ASD (mm)') + "\n")
            f.write("-" * (args.class_name_width + 40) + "\n") # 调整了分隔线长度

            for i, class_name in enumerate(CLASS_NAMES):
                # 从字典中安全地获取跨 seed 的均值和标准差
                col_name_dice = f"Dice_Class{i+1}"
                dice_mean = per_class_means_across_seeds.get("Dice", {}).get(col_name_dice, np.nan) * 100
                dice_std = per_class_stds_across_seeds.get("Dice", {}).get(col_name_dice, np.nan) * 100

                col_name_asd = f"ASD_Class{i+1}"
                asd_mean = per_class_means_across_seeds.get("ASD", {}).get(col_name_asd, np.nan)
                asd_std = per_class_stds_across_seeds.get("ASD", {}).get(col_name_asd, np.nan)

                dice_str = f"{dice_mean:5.2f} ± {dice_std:5.2f}" if not np.isnan(dice_mean) else "N/A"
                asd_str = f"{asd_mean:5.3f} ± {asd_std:5.3f}" if not np.isnan(asd_mean) else "N/A"

                # --- 修改了 f.write ---
                f.write(row_format.format(class_name, dice_str, asd_str) + "\n")
            f.write("-" * (args.class_name_width + 40) + "\n\n") # 调整了分隔线长度

            # --- Overall & Tail Averages (使用新的计算结果) ---
            f.write("🔷 Overall & Tail Average Metrics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric':<22} | {'Value (Mean ± Std)'}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Avg Foreground Dice':<22} | {final_results.get('Avg_FG_Dice', 'N/A')}\n")
            f.write(f"{'Avg Foreground ASD':<22} | {final_results.get('Avg_FG_ASD', 'N/A')}\n")
            f.write("-" * 50 + "\n")
            if tail_indices:
                f.write(f"{'Avg Tail Dice':<22} | {final_results.get('Avg_Tail_Dice', 'N/A')}\n")
                f.write(f"{'Avg Tail ASD':<22} | {final_results.get('Avg_Tail_ASD', 'N/A')}\n")
                f.write("-" * 50 + "\n")
                f.write("\n📌 Tail Classes: " + ", ".join(TAIL_CLASSES) + "\n")

        print(f"✅ Final summary results saved to: {args.summary_output}")

    except Exception as e:
        print(f"\n❌ Error writing summary file '{args.summary_output}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # --- 修改了描述 ---
        description="Aggregate per-sample results from multiple seeds found in a directory structure. Calculates Mean±Std (Dice, ASD) for tables and prepares aggregated per-sample data for p-value testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 输入参数 ---
    parser.add_argument('--method_base_dir', type=str, required=True,
                        help='Path to the base directory of the method (e.g., ./logs/amos/cps/). Contains subdirectories for each seed.')
    parser.add_argument('--seed_pattern', type=str, default='seed_*',
                        help='Pattern to find seed directories within the base directory (e.g., seed_*, run_*).')
    parser.add_argument('--results_filename', type=str, default='result',
                        help='Base name of the result file (WITHOUT extension) within each seed directory (e.g., result for result.csv or result.xlsx).')

    # --- 输出参数 ---
    parser.add_argument('--aggregated_csv_output', type=str, required=True,
                        help='Path to save the NEW CSV containing per-sample results averaged across seeds (input for p-value script).')
    parser.add_argument('--summary_output', type=str, default='aggregated_summary.txt',
                        help='Path to save the final Mean ± Std summary text file (for paper table).')

    # --- 控制参数 ---
    parser.add.argument('--min_seeds', type=int, default=2,
                        help='Minimum number of valid seed result files required to proceed.')
    parser.add_argument('--class_name_width', type=int, default=22,
                         help='Width for the class name column in the summary output file for alignment.')

    args = parser.parse_args()

    # --- 运行主函数 ---
    main(args)