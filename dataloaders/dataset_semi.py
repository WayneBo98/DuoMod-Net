import os
import numpy as np
import itertools
from torch.utils.data import Sampler
from monai.data import Dataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandFlipd,
    ToTensorD,
    SpatialPadD
)
import torch

# ==============================================================================
# 1. 粘贴您提供的 Sampler 及其辅助函数
# ==============================================================================
def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            # 注意这里应该是 list(primary_batch) + list(secondary_batch)
            # 否则会返回一个元组列表而不是一个拼接的列表
            list(primary_batch) + list(secondary_batch)
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

# ==============================================================================
# 2. 定义路径和参数
# ==============================================================================
data_dir = "/data/wangbo/CissMOS/Datasets/Amos22"
labeled_list_file = "/data/wangbo/SSL_imbalance/amos_data/splits/labeled_5p.txt"
unlabeled_list_file = "/data/wangbo/SSL_imbalance/amos_data/splits/unlabeled_5p.txt"
valid_list_file = "/data/wangbo/SSL_imbalance/amos_data/splits/eval.txt" # 验证集部分保持不变

# ==============================================================================
# 3. 准备合并的数据列表和索引
# ==============================================================================
def load_data_dicts(file_list_path, data_dir, is_labeled=True):
    with open(file_list_path, 'r') as f:
        base_filenames = [line.strip() for line in f.readlines() if line.strip()]
    data_dicts = []
    for base_name in base_filenames:
        data_dict = {"image": os.path.join(data_dir, "imagesTr", f"{base_name}.nii.gz")}
        if is_labeled:
            data_dict["label"] = os.path.join(data_dir, "labelsTr", f"{base_name}.nii.gz")
        data_dicts.append(data_dict)
    return data_dicts

# 加载文件列表
labeled_files = load_data_dicts(labeled_list_file, data_dir, is_labeled=True)
unlabeled_files = load_data_dicts(unlabeled_list_file, data_dir, is_labeled=False)
val_files = load_data_dicts(valid_list_file, data_dir, is_labeled=True) # 验证集不变

# **关键步骤：合并数据集并创建索引**
combined_files = labeled_files + unlabeled_files
labeled_indices = list(range(len(labeled_files)))
unlabeled_indices = list(range(len(labeled_files), len(combined_files)))

print(f"合并后总训练样本数: {len(combined_files)}")
print(f"其中有标签样本索引范围: 0 - {len(labeled_indices)-1}")
print(f"其中无标签样本索引范围: {len(labeled_indices)} - {len(combined_files)-1}")

# ==============================================================================
# 4. 定义统一的、智能的变换 (Transforms)
# ==============================================================================
# 由于一个变换流需要处理两种数据，我们需要让它足够智能
# 特别是对于裁剪，有标签和无标签的策略不同
class SmartCropd(object):
    """
    一个智能裁剪类：
    - 如果数据字典中存在 'label' 键，则使用 RandCropByPosNegLabeld
    - 否则，使用 RandSpatialCropd
    """
    def __init__(self, spatial_size, label_key='label', num_samples=1):
        self.label_cropper = RandCropByPosNegLabeld(
            keys=["image", "label"], label_key=label_key, spatial_size=spatial_size,
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0
        )
        self.spatial_cropper = RandSpatialCropd(
            keys=["image"], roi_size=spatial_size, random_size=False
        )
    
    def __call__(self, data):
        if 'label' in data and data['label'] is not None:
            # `label_cropper` 返回的是一个列表，我们从列表中取出唯一的元素（字典）
            return self.label_cropper(data)[0]  # <--- 关键修正在这里！
        else:
            return self.spatial_cropper(data)

# 创建统一的训练变换流
train_transforms = Compose(
    [
        # allow_missing_keys=True 至关重要，它允许label文件不存在
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        # 对标签也允许缺失，因为它可能不存在
        EnsureChannelFirstd(keys=["label"], allow_missing_keys=True, channel_dim='no_channel'),
        Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"), allow_missing_keys=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True),
        SpatialPadD(
            keys=["image", "label"],
            spatial_size=(96, 96, 96),
            method="end",  # "end" 表示在图像的右/下/后方进行填充
            allow_missing_keys=True
        ),
        # --- 使用我们自定义的智能裁剪 ---
        SmartCropd(spatial_size=(96, 96, 96), num_samples=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1), allow_missing_keys=True),
        ToTensorD(keys=["image", "label"], allow_missing_keys=True),
    ]
)

# 验证集变换保持不变
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensorD(keys=["image", "label"]),
    ]
)

def semi_supervised_collate(batch):
    """
    自定义 Collate 函数，用于处理半监督学习中的混合批次。
    `batch` 是一个列表，列表元素是 `Dataset` 返回的字典。
    """
    # 初始化一个空字典来存放最终的批次数据
    collated_batch = {}
    
    # 收集所有样本中出现过的所有键
    all_keys = set().union(*[d.keys() for d in batch])
    
    for key in all_keys:
        # 收集这个批次中所有包含当前键的样本的对应值
        # 这是一个列表，里面是每个样本的 image tensor 或 label tensor
        existing_values = [d[key] for d in batch if key in d and d[key] is not None]
        
        # 如果这个键在任何样本中都没有有效值，则跳过
        if not existing_values:
            continue
            
        # 使用 MONAI 的默认 collate 函数来堆叠这些有效值
        # 这会自动处理张量堆叠、元数据合并等
        collated_batch[key] = list_data_collate(existing_values)
        
    return collated_batch

# ==============================================================================
# 5. 创建单个 Dataset 和 使用了 TwoStreamBatchSampler 的 DataLoader
# ==============================================================================
# 定义批次大小
TOTAL_BATCH_SIZE = 6
LABELED_BATCH_SIZE = 2 # 每个批次中包含2个有标签样本

# 创建统一的数据集
combined_ds = Dataset(data=combined_files, transform=train_transforms)

# 创建采样器实例，让无标签数据作为主数据流
batch_sampler = TwoStreamBatchSampler(
    primary_indices=unlabeled_indices,
    secondary_indices=labeled_indices,
    batch_size=TOTAL_BATCH_SIZE,
    secondary_batch_size=LABELED_BATCH_SIZE
)

# **关键：创建单个 DataLoader，并传入 batch_sampler**
# 注意：当使用 batch_sampler 时, batch_size, shuffle, sampler, drop_last 都必须为默认值
train_loader = DataLoader(
    combined_ds,
    batch_sampler=batch_sampler,
    num_workers=4,
    collate_fn=semi_supervised_collate
    # collate_fn=pad_list_data_collate # 如果patch大小不一或有缺失键，可能需要自定义collate
)

# 验证集加载器保持不变
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)


# ==============================================================================
# 6. 迭代检查新的加载器
# ==============================================================================
print("\n--- 检查使用 TwoStreamBatchSampler 的加载器输出 ---")

# 计算每个epoch的步数
num_steps_per_epoch = len(batch_sampler)
print(f"每个 epoch 将进行 {num_steps_per_epoch} 步。")

# 训练循环现在变得非常简洁
for i, batch_data in enumerate(train_loader):
    print(f"Step {i+1}/{num_steps_per_epoch}:")
    print(f"  混合批次 'image' shape: {batch_data['image'].shape}")
    # 检查标签是否存在，以及它的shape
    if 'label' in batch_data:
        # 由于collate默认行为，无标签样本的label可能是None或者被过滤掉
        # 实际批次中的标签张量可能只包含有标签样本的标签
        print(f"  混合批次 'label' shape: {batch_data['label'].shape}")
    else:
        print("  混合批次中没有 'label' 键。")

    if i >= 2: # 只演示前3步
        break