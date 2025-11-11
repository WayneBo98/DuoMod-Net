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
    
    print(f"Merging labels 16 -> 15 for {os.path.basename(label_path)}")
    label_np[label_np == 16] = 15
    
    # 将修改后的 NumPy 数组转换回 SimpleITK 图像，以便保存为 .nii.gz 格式
    # 这一步很关键，因为它创建了一个新的 SimpleITK 对象
    modified_resampled_label = sitk.GetImageFromArray(label_np)
    
    # 必须将原始重采样标签的元数据（spacing, origin, direction）复制过来
    # 否则保存的 nifti 文件会丢失空间信息
    modified_resampled_label.SetSpacing(resampled_label.GetSpacing())
    modified_resampled_label.SetOrigin(resampled_label.GetOrigin())
    modified_resampled_label.SetDirection(resampled_label.GetDirection())
    # =======================================================
    
    # 返回修改后的标签对象和数组
    return resampled_image, modified_resampled_label, image_np, label_np
    # return resampled_image,resampled_label,image_np, label_np


def process(base_dir,save_dir_npy,save_dir_nii,target_spacing=(1.5, 1.5, 2.0)):

    for tag in ['Tr', 'Val', 'Ts']:
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

if __name__ == '__main__':
    base_dir = '/data/wangbo/CissMOS/Datasets/Word'
    save_dir_npy = '/data/wangbo/CissMOS/Datasets/Word_merge_npy'
    save_dir_nii = '/data/wangbo/CissMOS/Datasets/Word_merge_nii'
    os.makedirs(save_dir_npy, exist_ok=True)
    os.makedirs(save_dir_nii, exist_ok=True)
    for tag in ['Tr', 'Val', 'Ts']:
        os.makedirs(os.path.join(save_dir_npy,f'images{tag}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_npy,f'labels{tag}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_nii,f'images{tag}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir_nii,f'labels{tag}'), exist_ok=True)
    process(base_dir,save_dir_npy,save_dir_nii,target_spacing=(1.5, 1.5, 2.0))
