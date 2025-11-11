# DuoMod-Net: Rethinking Class Imbalanced Semi-supervised Medical Image Segmentation

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official implementation for our paper, "Rethinking Class Imbalanced Semi-supervised Medical Image Segmentation". We propose the Duo-component Modulation Network (DuoMod-Net) , a synergistic learning framework designed to tackle the dual challenges of class imbalance and sparse features in semi-supervised 3D medical image segmentation. We demonstrate its effectiveness on challenging public benchmarks, including AMOS , WORD , and FLARE22.

## 🔧 1. Setup

### 1.1. Clone Repository

```bash
git clone [https://github.com/WayneBo98/DuoMod-Net.git](https://github.com/WayneBo98/DuoMod-Net.git)
cd DuoMod-Net

### 1.2. Environment Setup

建议使用 `conda` 或 `venv` 创建虚拟环境。

```bash
# 使用 conda (推荐)
conda create -n amos_env python=3.9
conda activate amos_env

# 安装 PyTorch (请根据您的 CUDA 版本
# 访问 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

建议在 `requirements.txt` 文件中包含以下核心库：

```text
numpy
pandas
SimpleITK
tqdm
openpyxl
scipy
# nibabel (如果预处理需要)
# ...
```

## 📁 2. Dataset Preparation (AMOS 2022)

### 2.1. Download Data

1.  从 AMOS 2022 官网下载数据集：[https://amos-challenge.grand-challenge.org/](https://amos-challenge.grand-challenge.org/)
2.  下载 "Task 1: Abdominal partial CT" (AMOS 22) 部分。

### 2.2. Directory Structure

将下载的数据解压并组织成以下推荐的目录结构。**请将 `Dataset` 目录放在项目根目录下。**

```
[Your-Repo-Root]/
├── Dataset/
│   ├── AMOS22/
│   │   ├── imagesTr/           # 500 个训练图像 (e.g., amos_0001.nii.gz)
│   │   ├── labelsTr/           # 500 个训练标签
│   │   ├── imagesVa/           # 100 个验证图像
│   │   ├── labelsVa/           # 100 个验证标签
│   │   ├── imagesTs/           # (公开测试集，如果使用)
│   │   └── dataset.json        # (数据集元数据)
│
├── logs/                       # 用于存放训练日志、模型和结果
├── ... (其他代码目录)
├── train.py
├── test.py                     # (您之前的 test_word.py)
├── eval_amos.py                # (您的评估脚本)
├── aggregate_results.py        # (您的聚合脚本)
├── run_unified_pipeline.sh     # (您的统一运行脚本)
└── README.md
```

### 2.3. Preprocessing

我们的模型需要 `.npy` 格式的预处理数据以加快 I/O。请运行预处理脚本（**您需要提供或修改此脚本**）。

```bash
# 这是一个示例命令，请根据您的 preprocess.py 进行修改
python preprocess.py \
    --data_path ./Dataset/AMOS22 \
    --output_path ./Dataset/AMOS22_preprocessed \
    --num_classes 16
```

预处理完成后，您的 `Dataset` 目录应该如下所示：

```
Dataset/
├── AMOS22/
│   ├── imagesTr/
│   ├── ...
├── AMOS22_preprocessed/
│   ├── imagesTr/               # (e.g., amos_0001.npy)
│   ├── labelsTr/
│   ├── imagesVa/
│   ├── labelsVa/
│   ├── imagesTs/
│   └── ...
```

## 🚀 3. How to Run

我们提供了一个统一的 Shell 脚本来管理推理、评估和结果聚合。

### 3.1. Training (假设)

首先，您需要训练模型。以下是一个多 SEED 训练的示例脚本（**请根据您的 `train.py` 调整参数**）。

```bash
# 示例：训练 3 个 seeds (0, 1, 2)
GPU_ID=0

for seed in 0 1 2; do
    echo "--- Training SEED ${seed} ---"
    
    python train.py \
        --model "slcnet" \
        --data_path ./Dataset/AMOS22_preprocessed \
        --output_dir ./logs/amos/slcnet/seed_${seed} \
        --seed ${seed} \
        --gpu "${GPU_ID}" \
        --num_classes 16 \
        --max_epochs 1000 \
        --batch_size 2
        # ... (添加其他训练参数)
        
    echo "--- SEED ${seed} Training Done ---"
done
```

训练完成后，模型（如 `best_model.pth`）应保存在 `logs/amos/slcnet/seed_X/ckpts/` 目录下。

### 3.2. Unified Inference & Evaluation

我们强烈建议使用 `run_unified_pipeline.sh` 脚本来执行完整的**推理、评估和结果汇总**流程。

**Step 1: 配置 Pipeline**

打开 `run_unified_pipeline.sh` (这是我们之前合并的脚本)，修改 `METHODS_TO_PROCESS` 数组，指定您训练好的模型和数据路径。

```bash
# run_unified_pipeline.sh

# ... (通用设置)
GT_PATH="./Dataset/AMOS22/labelsVa" # 假设在验证集上评估
NUM_CLASSES=16 # 15 个前景 + 1 个背景
SEEDS=(0 1 2)
GPU_ID="0"
# ...

# --- D. 方法配置 ---
METHODS_TO_PROCESS=(
    # "方法名称;日志根目录;NPY数据路径;NII数据路径;任务名称"
    
    # 示例: 评估您刚训练的 slcnet
    "slcnet;./logs/amos/slcnet;./Dataset/AMOS22_preprocessed/imagesVa;./Dataset/AMOS22/imagesVa;amos"
    
    # 示例: 评估另一个基线
    # "baseline;./logs/amos/baseline;./Dataset/AMOS22_preprocessed/imagesVa;./Dataset/AMOS22/imagesVa;amos"
)
```

**Step 2: 运行 Pipeline**

确保您已设置好流程控制开关（`RUN_INFERENCE`, `RUN_EVALUATION`, `RUN_AGGREGATION`）。

```bash
bash run_unified_pipeline.sh
```

**Step 3: 查看结果**

脚本执行完毕后，您将获得两个关键产物：

1.  **最终表格 (Mean ± Std):**
    * 路径: `logs/amos/slcnet/slcnet_summary_results.txt`
    * 内容: 包含 Dice 和 ASD 的均值与标准差，可直接用于论文。

2.  **p-value 计算数据:**
    * 路径: `logs/amos/slcnet/slcnet_aggregated_metrics.csv`
    * 内容: 包含每个**样本**跨 seed 的平均指标，用于后续的统计显著性检验（如 T 检验）。

## (可选) 4. Manual Workflow

如果您想分步执行，也可以手动调用 Python 脚本。

### 4.1. Manual Inference

(使用您的 `test.py` 脚本)

```bash
python test.py \
    --npy_path ./Dataset/AMOS22_preprocessed/imagesVa \
    --original_nii_path ./Dataset/AMOS22/imagesVa \
    --output_path ./logs/amos/slcnet/seed_0/predictions \
    --model_path ./logs/amos/slcnet/seed_0/ckpts/best_model.pth \
    --gpu "0" \
    --exp "slcnet" \
    --task "amos" \
    --overlap 0.5
```

### 4.2. Manual Evaluation

(使用您的 `eval_amos.py` 脚本，注意我们已移除 HD95)

```bash
python eval_amos.py \
    --pred_path ./logs/amos/slcnet/seed_0/predictions \
    --gt_path ./Dataset/AMOS22/labelsVa \
    --num_classes 16 \
    --output_csv ./logs/amos/slcnet/seed_0/result.csv
```

### 4.3. Manual Aggregation

(使用您的 `aggregate_results.py` 脚本，注意我们已移除 HD95)

```bash
python aggregate_results.py \
    --method_base_dir ./logs/amos/slcnet \
    --seed_pattern "seed_*" \
    --results_filename "result" \
    --aggregated_csv_output ./logs/amos/slcnet/slcnet_aggregated_metrics.csv \
    --summary_output ./logs/amos/slcnet/slcnet_summary_results.txt
```

## 📊 Example Results

运行 `run_unified_pipeline.sh` 后，您将在 `summary_output` 文件中看到类似以下的格式化结果：

```text
📊 Aggregated Summary Results (Mean ± Std from 3 seeds)
Method Base Directory: ./logs/amos/slcnet
======================================================================

🔷 Per-Class Metrics:
--------------------------------------------------
Class                  |           Dice (%) |            ASD (mm)
----------------------------------------------------------------
spleen                 |     95.12 ±  0.30  |     0.512 ±  0.101
right kidney           |     94.00 ±  0.50  |     0.600 ±  0.120
left kidney            |     93.50 ±  0.45  |     0.650 ±  0.110
... (其他类别)
--------------------------------------------------

🔷 Overall & Tail Average Metrics:
--------------------------------------------------
Metric                 | Value (Mean ± Std)
--------------------------------------------------
Avg Foreground Dice    | 90.50 ± 0.80
Avg Foreground ASD     | 1.200 ± 0.300
--------------------------------------------------
Avg Tail Dice          | 85.10 ± 1.10
Avg Tail ASD           | 2.100 ± 0.500
--------------------------------------------------
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

* Our code is built upon [mention any baseline frameworks, e.g., nnU-Net, PyTorch].
* We thank the organizers of the [AMOS 2022 Challenge](https://amos-challenge.grand-challenge.org/) for providing the dataset.
