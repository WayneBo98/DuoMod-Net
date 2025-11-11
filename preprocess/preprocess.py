import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
sys.path.append('/data/wangbo/SSL_imbalance/code/')
from utils import read_list, read_nifti
import SimpleITK as sitk


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')
            
def resample_to_spacing(image_path, label_path, target_spacing=(1.5, 1.5, 2.0)):
    # 读取图像和标签
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    label = sitk.ReadImage(label_path, sitk.sitkInt8)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算重采样后的新尺寸
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    # 创建重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    # 对图像进行线性插值
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(image)

    # 对标签进行最近邻插值，以保证标签值不改变
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_label = resampler.Execute(label)

    # 转换为 NumPy 数组
    image_np = sitk.GetArrayFromImage(resampled_image) # shape: (D, H, W)
    label_np = sitk.GetArrayFromImage(resampled_label) # shape: (D, H, W)
    
    return resampled_image,resampled_label,image_np, label_np


def process(base_dir,save_dir_npy,save_dir_nii,target_spacing=(1.5, 1.5, 2.0)):

    for tag in ['Tr', 'Va']:
    # for tag in ['Tr']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, f'images{tag}', '*.nii.gz'))):
            img_id = path.split('/')[-1].split('.')[0]
            img_ids.append(img_id)

            image_path = os.path.join(base_dir, f'images{tag}', f'{img_id}.nii.gz')
            label_path =os.path.join(base_dir, f'labels{tag}', f'{img_id}.nii.gz')
            
            resample_image,resample_label,image, label = resample_to_spacing(image_path, label_path, target_spacing=target_spacing)
            sitk.WriteImage(resample_image, os.path.join(save_dir_nii,f'images{tag}', f'{img_id}.nii.gz'))
            sitk.WriteImage(resample_label, os.path.join(save_dir_nii,f'labels{tag}', f'{img_id}.nii.gz'))
            np.save(os.path.join(save_dir_npy,f'images{tag}', f'{img_id}_image.npy'),image)
            np.save(os.path.join(save_dir_npy,f'labels{tag}', f'{img_id}_label.npy'),label)

# def process_split_fully(train_ratio=0.9):
#     if not os.path.exists(os.path.join(config.save_dir, 'splits')):
#         os.makedirs(os.path.join(config.save_dir, 'splits'))
#     for tag in ['Tr', 'Va']:
#         img_ids = []
#         for path in tqdm(glob.glob(os.path.join(config.base_dir, f'images{tag}', '*.nii.gz'))):
#             img_id = path.split('/')[-1].split('.')[0]
#             img_ids.append(img_id)
#         print(img_ids)
        
#         if tag == 'Tr':
#             train_val_ids = np.random.permutation(img_ids)
#             # split_idx = int(len(img_ids) * train_ratio)
#             # train_val_ids = img_ids[:split_idx]
#             # test_ids = sorted(img_ids[split_idx:])

#             # train_val_ids = [i for i in img_ids if i not in test_ids]
#             split_idx = int(len(train_val_ids) * train_ratio)
#             train_ids = sorted(train_val_ids[:split_idx])
#             eval_ids = sorted(train_val_ids[split_idx:])
#             write_txt(
#                 train_ids,
#                 os.path.join(config.save_dir, 'splits/train.txt')
#             )
#             write_txt(
#                 eval_ids,
#                 os.path.join(config.save_dir, 'splits/eval.txt')
#             )

#         else:
#             test_ids = np.random.permutation(img_ids)
#             test_ids = sorted(test_ids)
#             write_txt(
#                 test_ids,
#                 os.path.join(config.save_dir, 'splits/test.txt')
#             )


# def process_split_semi(split='train', labeled_ratio=10):
#     ids_list = read_list(split, task="amos")
#     ids_list = np.random.permutation(ids_list)

#     split_idx = int(len(ids_list) * labeled_ratio/100)
#     labeled_ids = sorted(ids_list[:split_idx])
#     unlabeled_ids = sorted(ids_list[split_idx:])
    
#     write_txt(
#         labeled_ids,
#         os.path.join(config.save_dir, f'splits/labeled_{labeled_ratio}p.txt')
#     )
#     write_txt(
#         unlabeled_ids,
#         os.path.join(config.save_dir, f'splits/unlabeled_{labeled_ratio}p.txt')
#     )


if __name__ == '__main__':
    base_dir = '/data/wangbo/CissMOS/Datasets/Amos22'
    save_dir_npy = '/data/wangbo/CissMOS/Datasets/AMOS22_1.5_2.0_npy'
    save_dir_nii = '/data/wangbo/CissMOS/Datasets/AMOS22_1.5_2.0_nii'
    os.makedirs(save_dir_npy, exist_ok=True)
    os.makedirs(save_dir_nii, exist_ok=True)
    for tag in ['Tr', 'Va']:
        os.makedirs(os.path.join(save_dir_npy,f'images{tag}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_npy,f'labels{tag}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_nii,f'images{tag}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_nii,f'labels{tag}'), exist_ok=True)
    process(base_dir,save_dir_npy,save_dir_nii,target_spacing=(1.5, 1.5, 2.0))
    # process_npy()
    # process_split_fully()
    # process_split_semi(labeled_ratio=2)
    # process_split_semi(labeled_ratio=5)
    # process_split_semi(labeled_ratio=10)
