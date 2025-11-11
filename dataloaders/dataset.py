import os
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    ToTensorD,
    # 注意：Spacingd, Orientationd, CropForegroundd 已被移除
)
import torch

# ==============================================================================
# 1. 定义路径和参数
# ==============================================================================
# 【修改】数据集根目录现在指向预处理后的数据
data_dir = "/data/wangbo/CissMOS/Datasets/Amos22_1.5_2.0" 
# 文件列表路径保持不变
train_list_file = "/data/wangbo/SSL_imbalance/amos_data/splits/train.txt"
valid_list_file = "/data/wangbo/SSL_imbalance/amos_data/splits/eval.txt"

# ==============================================================================
# 2. 定义数据列表加载函数
# ==============================================================================
def load_data_dicts_from_file(file_list_path, data_dir):
    """
    从 .txt 文件加载基础文件名，并构建MONAI所需的数据字典列表。
    这个函数无需修改，因为它能正确地在新的预处理目录中找到文件。
    """
    with open(file_list_path, 'r') as f:
        base_filenames = [line.strip() for line in f.readlines() if line.strip()]

    data_dicts = [
        {
            "image": os.path.join(data_dir, "imagesTr", f"{base_name}.nii.gz"),
            "label": os.path.join(data_dir, "labelsTr", f"{base_name}.nii.gz")
        }
        for base_name in base_filenames
    ]
    return data_dicts

# ==============================================================================
# 3. 加载训练和验证数据列表
# ==============================================================================
train_files = load_data_dicts_from_file(train_list_file, data_dir)
val_files = load_data_dicts_from_file(valid_list_file, data_dir)

print(f"成功加载训练样本数量: {len(train_files)}")
print(f"成功加载验证样本数量: {len(val_files)}")
print("\n训练样本示例 (来自预处理目录):")
print(train_files[0])
print("\n验证样本示例 (来自预处理目录):")
print(val_files[0])

# ==============================================================================
# 4. 定义训练和验证集的变换 (Transforms)
# ==============================================================================

# 【修改】训练集变换：移除了预处理步骤，只保留归一化和数据增强
train_transforms = Compose(
    [
        # 仍然需要加载数据和确保通道维度
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # --- 以下变换已被预处理脚本完成，故移除 ---
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # 保留强度归一化，这是训练时常见的步骤
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        
        # --- 数据增强部分保持不变 ---
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 64),
            pos=1, neg=1,
            num_samples=4, # 每个原始图像生成4个训练patch
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1)), # 左右和前后翻转
        ToTensorD(keys=["image", "label"]),
    ]
)

# 【修改】验证集变换：同样移除了预处理步骤
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ToTensorD(keys=["image", "label"]),
    ]
)

# ==============================================================================
# 5. 创建 MONAI Dataset 和 DataLoader
# ==============================================================================

# 训练集
train_ds = Dataset(data=train_files, transform=train_transforms)
# batch_size=2，因为num_samples=4，所以每个batch实际包含 2*4=8 个patch
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

# 验证集
val_ds = Dataset(data=val_files, transform=val_transforms)
# 验证时通常batch_size为1
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# ==============================================================================
# 6. 迭代检查输出
# ==============================================================================
print("\n--- 检查数据加载器输出 ---")
try:
    # 检查一个训练批次
    train_batch = next(iter(train_loader))
    print(f"训练批次 'image' shape: {train_batch['image'].shape}")
    print(f"训练批次 'label' shape: {train_batch['label'].shape}")

    # 检查一个验证批次
    val_batch = next(iter(val_loader))
    # 注意：验证集的shape现在应该是预处理后裁剪的大小
    print(f"\n验证批次 'image' shape: {val_batch['image'].shape}")
    print(f"验证批次 'label' shape: {val_batch['label'].shape}")
except Exception as e:
    print(f"检查数据加载时发生错误: {e}")
    print("请确保您的路径和文件列表正确无误。")

