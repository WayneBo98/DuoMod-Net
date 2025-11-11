#!/usr/bin/env python3
# eval_metrics.py - 计算 Dice, ASD 并保存每个样本的结果到 CSV

import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd # 引入 Pandas 用于 CSV 操作

# --- (calculate_dice_score 函数保持不变) ---
def calculate_dice_score(pred_path, gt_path, num_classes):
    """计算 Dice 分数 (代码与之前相同)"""
    try:
        pred_sitk = sitk.ReadImage(pred_path)
        pred_array = sitk.GetArrayFromImage(pred_sitk)
        gt_sitk = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_sitk)
    except Exception as e:
        print(f"Error loading files {pred_path} or {gt_path}: {e}")
        return [np.nan] * (num_classes - 1) # 返回 NaN 列表

    if pred_array.shape != gt_array.shape:
        print(f"Shape mismatch: Pred {pred_array.shape} vs GT {gt_array.shape} for {os.path.basename(pred_path)}")
        return [np.nan] * (num_classes - 1)

    dices = []
    for c in range(1, num_classes):
        pred_mask = (pred_array == c)
        gt_mask = (gt_array == c)
        if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
            dice = 1.0
        else:
            intersection = np.sum(pred_mask & gt_mask)
            denominator = np.sum(pred_mask) + np.sum(gt_mask)
            if denominator == 0:
                 # 一个为空一个非空的情况
                dice = 0.0
            else:
                dice = (2.0 * intersection) / denominator
        dices.append(dice)
    return dices


# --- (修改后的 calculate_asd 函数) ---
def calculate_asd(pred_path, gt_path, num_classes, spacing=None):
    """
    计算 ASD (平均表面距离)。
    返回: list of asd values [asd_c1, asd_c2, ...]
          如果无法计算，返回 nan。
    """
    nan_result = float('nan')
    results = [nan_result] * (num_classes - 1) # 初始化为 NaN

    try:
        pred_sitk = sitk.ReadImage(pred_path)
        pred_array = sitk.GetArrayFromImage(pred_sitk)
        gt_sitk = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_sitk)

        # 确保从图像读取正确的 spacing (x, y, z for SimpleITK)
        if spacing is None:
            spacing_xyz = pred_sitk.GetSpacing()
            if not spacing_xyz or len(spacing_xyz) != 3:
                raise ValueError("Could not read valid spacing from image.")
        else:
            spacing_xyz = spacing # 假设传入的就是 (x, y, z)

    except Exception as e:
        print(f"Error loading files/spacing for ASD {os.path.basename(pred_path)}: {e}")
        return results # 返回全 NaN 列表

    if pred_array.shape != gt_array.shape:
        print(f"Shape mismatch: {os.path.basename(pred_path)}")
        return results

    # 针对每个前景类别计算
    for c in range(1, num_classes):
        pred_mask = (pred_array == c).astype(np.uint8)
        gt_mask = (gt_array == c).astype(np.uint8)

        pred_empty = np.sum(pred_mask) == 0
        gt_empty = np.sum(gt_mask) == 0

        if pred_empty and gt_empty:
            results[c-1] = 0.0 # Correct case: both empty, distance is 0
            continue
        elif pred_empty or gt_empty:
            # print(f"Warning: Class {c} empty in pred or GT for {os.path.basename(pred_path)}. ASD set to NaN.")
            results[c-1] = nan_result # One empty, one not: Cannot compute distance reliably
            continue

        pred_mask_sitk = sitk.GetImageFromArray(pred_mask)
        gt_mask_sitk = sitk.GetImageFromArray(gt_mask)
        pred_mask_sitk.SetSpacing(spacing_xyz)
        gt_mask_sitk.SetSpacing(spacing_xyz)

        try:
            # 使用 HausdorffDistanceImageFilter 计算 ASD
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(pred_mask_sitk, gt_mask_sitk)
            asd = hausdorff_distance_filter.GetAverageHausdorffDistance()

            results[c-1] = asd

        except Exception as e:
            print(f"Error computing ASD for class {c} in {os.path.basename(pred_path)}: {e}")
            results[c-1] = nan_result # Keep as NaN on error

    return results

def main(args):
    try:
        pred_files = sorted([f for f in os.listdir(args.pred_path) if f.endswith(('.nii.gz', '.nii'))])
        gt_files_in_dir = sorted([f for f in os.listdir(args.gt_path) if f.endswith(('.nii.gz', '.nii'))])
    except FileNotFoundError as e:
        print(f"Error: Directory not found - {e.strerror}: {e.filename}")
        return

    if not pred_files or not gt_files_in_dir:
        print("Error: One of the directories is empty or contains no compatible files.")
        return

    # --- File Matching Logic ---
    gt_map = {f: os.path.join(args.gt_path, f) for f in gt_files_in_dir}
    evaluation_pairs = []
    for pred_file in pred_files:
        if pred_file in gt_map:
            evaluation_pairs.append((pred_file, gt_map[pred_file]))
        else:
            print(f"Warning: Ground truth for prediction '{pred_file}' not found. Skipping.")

    if not evaluation_pairs:
        print("Error: No matching prediction/ground truth file pairs found.")
        return

    num_foreground_classes = args.num_classes - 1
    # 使用列表存储每个样本的结果字典
    results_list = []

    print(f"🚀 Starting Dice & ASD evaluation for {len(evaluation_pairs)} file pairs...")

    for pred_filename, gt_filepath in tqdm(evaluation_pairs, desc="Evaluating"):
        pred_filepath = os.path.join(args.pred_path, pred_filename)
        sample_results = {"Filename": pred_filename} # 初始化当前样本的字典

        # 尝试读取 spacing (x, y, z)
        current_spacing_xyz = None
        try:
            img = sitk.ReadImage(pred_filepath)
            current_spacing_xyz = img.GetSpacing()
            if not current_spacing_xyz or len(current_spacing_xyz) != 3:
                raise ValueError("Invalid spacing tuple")
        except Exception as e:
            print(f"Warning: Cannot read spacing for {pred_filename}, skipping surface distances. Error: {e}")

        # 计算 Dice
        dices = calculate_dice_score(pred_filepath, gt_filepath, args.num_classes)
        for i in range(num_foreground_classes):
            sample_results[f"Dice_Class{i+1}"] = dices[i]

        # 计算 ASD (只有在 spacing 有效时)
        if current_spacing_xyz:
            # 调用修改后的函数
            asds = calculate_asd(pred_filepath, gt_filepath, args.num_classes, spacing=current_spacing_xyz)
            for i in range(num_foreground_classes):
                asd = asds[i]
                sample_results[f"ASD_Class{i+1}"] = asd
        else:
             # 如果 spacing 无效，填充 NaN
             for i in range(num_foreground_classes):
                 sample_results[f"ASD_Class{i+1}"] = float('nan')

        results_list.append(sample_results)

    # --- 将结果列表转换为 Pandas DataFrame ---
    results_df = pd.DataFrame(results_list)

    # --- 保存到 CSV ---
    try:
        results_df.to_csv(args.output_csv, index=False, float_format='%.6f')
        print(f"\n✅ Per-sample results successfully saved to: {args.output_csv}")
    except Exception as e:
        print(f"\n❌ Error saving results to CSV: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Dice & ASD scores and save per-sample results to CSV.")
    parser.add_argument('--pred_path', type=str, default='/data/wangbo/CissMOS/logs/amos/cps/newloss/seed_2/predictions', help="Path to predicted segmentations.")
    parser.add_argument('--gt_path', type=str, default='/data/wangbo/CissMOS/Datasets/Amos22/labelsVa', help="Path to ground truth labels.")
    parser.add_argument('--num_classes', type=int, default=16, help="Total number of classes including background.")
    parser.add_argument('--output_csv', type=str,  default='/data/wangbo/CissMOS/logs/amos/cps/newloss/seed_2/result.csv', help="Path to save the per-sample results CSV file.")
    # parser.add_argument('--class_names', nargs='+', default=None, help="Optional: List of foreground class names for headers.") # 可选，用于生成更友好的列名

    args = parser.parse_args()

    if args.num_classes <= 1:
        print("Error: --num_classes must be greater than 1.")
    else:
        main(args)