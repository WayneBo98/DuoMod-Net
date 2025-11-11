# 您可以创建一个新的工具文件，例如 utils/uncertainty_sampler.py

import torch
import torch.nn.functional as F
import numpy as np
import os
import math
from tqdm import tqdm
import random

# 您可以创建一个新的工具文件，例如 utils/uncertainty_sampler.py

import torch
import torch.nn.functional as F
import numpy as np
import os
import math
from tqdm import tqdm
import random
from scipy.ndimage import binary_closing, binary_fill_holes, label

class UncertaintyMapGenerator:
    def __init__(self, model_A, model_B, unlabeled_dataset, cache_dir, 
                 downsample_factor=4, required_divisor=32, 
                 use_ema_smoothing=False, ema_momentum=0.99):
        self.model_A = model_A
        self.model_B = model_B
        self.unlabeled_dataset = unlabeled_dataset
        self.cache_dir = cache_dir
        self.downsample_factor = downsample_factor
        self.required_divisor = required_divisor # 例如U-Net需要32 (2^5)
        self.use_ema_smoothing = use_ema_smoothing
        self.ema_momentum = ema_momentum
        os.makedirs(cache_dir, exist_ok=True)
        self.body_mask_cache_dir = '/data/wangbo/CissMOS/Datasets/AMOS22_1.5_2.0_npy/bodymaskTr'

    def _pad_to_divisible(self, tensor):
        """动态填充Tensor，使其尺寸可以被整除"""
        input_shape = tensor.shape[2:]
        padded_shape = [math.ceil(s / self.required_divisor) * self.required_divisor for s in input_shape]

        pads_for_f_pad = []
        pads_for_cropping = [] 

        for i in range(len(input_shape) - 1, -1, -1):
            total_pad = padded_shape[i] - input_shape[i]
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pads_for_f_pad.extend([pad_before, pad_after])
            pads_for_cropping.extend([pad_before, pad_after])
        
        padded_tensor = F.pad(tensor, tuple(pads_for_f_pad), mode='constant', value=0)
        return padded_tensor, input_shape, list(reversed(pads_for_cropping))

    @torch.no_grad()
    def generate_maps(self, map_type='disagreement',use_body_mask=False):
        print(f"--- Updating uncertainty maps (EMA: {self.use_ema_smoothing}, Type: {map_type}) ---, use_body_mask: {use_body_mask} ---")
        # 记录模型原始状态
        model_A_is_training = self.model_A.training
        model_B_is_training = self.model_B.training
        
        try:
            # 切换到评估模式
            self.model_A.eval()
            self.model_B.eval()

            for i in tqdm(range(len(self.unlabeled_dataset)), desc="Generating uncertainty maps"):
                # ... 您原有的、完全正确的推理逻辑 ...
                # (从 sample = self.unlabeled_dataset... 到 np.save(...) 的所有代码都不需要改变)
                sample = self.unlabeled_dataset.get_full_image_and_id(i)
                case_id = sample['case_id']

                image = sample['image']
                original_shape = image.shape
                    
                image_tensor = torch.from_numpy(image).float().cuda().unsqueeze(0).unsqueeze(0)

                low_res_shape = [s // self.downsample_factor for s in original_shape]
                low_res_image = F.interpolate(image_tensor, size=low_res_shape, mode='trilinear', align_corners=False)

                padded_image, original_low_res_shape, pads = self._pad_to_divisible(low_res_image)
                
                logits_A = self.model_A(padded_image)
                logits_B = self.model_B(padded_image)

                d, h, w = original_low_res_shape
                pad_d_before, _, pad_h_before, _, pad_w_before, _ = pads

                logits_A = logits_A[:, :, pad_d_before:pad_d_before+d, pad_h_before:pad_h_before+h, pad_w_before:pad_w_before+w]
                logits_B = logits_B[:, :, pad_d_before:pad_d_before+d, pad_h_before:pad_h_before+h, pad_w_before:pad_w_before+w]

                if map_type == 'js_divergence':
                    # 将 logits 转换为概率分布
                    probs_A = torch.softmax(logits_A, dim=1)
                    probs_B = torch.softmax(logits_B, dim=1)

                    # 计算平均分布
                    mean_probs = 0.5 * (probs_A + probs_B)
                    
                    epsilon = 1e-10
                    kl_A = F.kl_div((mean_probs + epsilon).log(), probs_A, reduction='none').sum(dim=1)
                    kl_B = F.kl_div((mean_probs + epsilon).log(), probs_B, reduction='none').sum(dim=1)
                    
                    # JS 散度是两个 KL 散度的平均
                    uncertainty_map_low_res = 0.5 * (kl_A + kl_B)
                    heatmap = F.interpolate(uncertainty_map_low_res.unsqueeze(0), size=original_shape, mode='trilinear', align_corners=False)
                
                elif map_type == 'disagreement': # 保留旧的硬不一致性作为备选
                    pred_A = torch.argmax(logits_A, dim=1)
                    pred_B = torch.argmax(logits_B, dim=1)
                    uncertainty_map_low_res = (pred_A != pred_B).float()
                    heatmap = F.interpolate(uncertainty_map_low_res.unsqueeze(0), size=original_shape, mode='nearest')
                    
                heatmap_numpy = heatmap.squeeze().cpu().numpy()
                
                if use_body_mask:
                    body_mask_path = os.path.join(self.body_mask_cache_dir, f"{case_id}_image.npy")
                    body_mask = np.load(body_mask_path)
                    # body_mask = self._create_body_mask(image)
                    heatmap_numpy = heatmap_numpy * body_mask
                # 对连续的热力图进行归一化，使其值在0-1之间，便于后续采样
                current_map_numpy = (heatmap_numpy - heatmap_numpy.min()) / (heatmap_numpy.max() - heatmap_numpy.min() + 1e-8)
                
                if self.use_ema_smoothing:
                    # --- EMA 平滑模式 ---
                    save_path = os.path.join(self.cache_dir, f"{case_id}_{map_type}_ema.npy")
                    
                    if os.path.exists(save_path):
                        last_ema_map = np.load(save_path)
                        new_ema_map = self.ema_momentum * last_ema_map + (1 - self.ema_momentum) * current_map_numpy
                        np.save(save_path, new_ema_map)
                    else:
                        # 第一次，直接保存当前图
                        np.save(save_path, current_map_numpy)
                else:
                    # --- 快照模式 (您之前的逻辑) ---
                    save_path = os.path.join(self.cache_dir, f"{case_id}_{map_type}_snapshot.npy")
                    np.save(save_path, current_map_numpy)
                
        finally:
            # 无论前面发生什么，都确保在函数结束时将模型恢复到原始状态
            if model_A_is_training:
                self.model_A.train()
            if model_B_is_training:
                self.model_B.train()
            print("--- Finished updating maps, models set back to original training state ---")

# 同样可以放在 utils/uncertainty_sampler.py

class UncertaintyGuidedCrop(object):
    """
    一个更健壮的、用于无标签数据的不确定性引导裁剪Transform。
    
    实现了混合采样策略：以 uncertainty_ratio 的概率进行不确定性引导采样，
    否则进行完全随机采样。
    """
    def __init__(self, output_size, cache_dir, uncertainty_ratio=0.8, map_type='disagreement',use_ema_smoothing=False):
        # 初始化output_size
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = tuple(output_size)
            
        self.cache_dir = cache_dir
        self.uncertainty_ratio = uncertainty_ratio
        self.use_ema_smoothing = use_ema_smoothing
        assert map_type in ['disagreement', 'js_divergence'], "map_type must be 'disagreement' or 'js_divergence'"
        self.map_type = map_type

    def __call__(self, sample):
        # --- 步骤 1: 对sample中的所有3D数组进行统一Padding (修正Bug) ---
        original_shape = sample['image'].shape
        
        # 计算padding量
        padding_values = []
        for i in range(3):
            pad_needed = self.output_size[i] - original_shape[i]
            if pad_needed > 0:
                pad_before = pad_needed // 2
                pad_after = pad_needed - pad_before
                padding_values.append((pad_before, pad_after))
            else:
                padding_values.append((0, 0))
        
        # 对sample字典中所有3D数组应用padding
        padded_sample = {}
        for key, item in sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                padded_sample[key] = np.pad(item, padding_values, mode='constant', constant_values=0)
            else:
                padded_sample[key] = item # 非3D数据（如case_id）直接保留
        
        padded_shape = padded_sample['image'].shape
        case_id = padded_sample['case_id']
        
        # --- 步骤 2: 混合采样逻辑 (修正逻辑) ---
        center_coords = None
        perform_uncertainty_sampling = random.random() < self.uncertainty_ratio

        if perform_uncertainty_sampling:
            if self.use_ema_smoothing:
                filename = f"{case_id}_{self.map_type}_ema.npy"
            else:
                filename = f"{case_id}_{self.map_type}_snapshot.npy"
            heatmap_path = os.path.join(self.cache_dir, filename)
            if os.path.exists(heatmap_path):
                heatmap = np.load(heatmap_path)
                # 对热力图也应用同样的padding，确保坐标系统一
                padded_heatmap = np.pad(heatmap, padding_values, mode='constant', constant_values=0)
                
                is_binary_map = (not self.use_ema_smoothing) and (self.map_type == 'disagreement')
                
                if is_binary_map:
                    # --- 硬采样逻辑 ---
                    indices = np.argwhere(padded_heatmap > 0)
                    if len(indices) > 0:
                        center_coords = random.choice(indices)
                else:
                    # --- 软采样逻辑 (适用于 js_divergence 和所有EMA平滑图) ---
                    flat_heatmap = padded_heatmap.flatten()
                    if flat_heatmap.sum() > 1e-6:
                        probs = flat_heatmap / flat_heatmap.sum()
                        chosen_index = np.random.choice(len(flat_heatmap), p=probs)
                        center_coords = np.unravel_index(chosen_index, padded_shape)

        # 如果决定进行随机采样，或者不确定性采样因故失败，则执行随机采样
        if center_coords is None:
            d1 = np.random.randint(0, padded_shape[0] - self.output_size[0] + 1)
            h1 = np.random.randint(0, padded_shape[1] - self.output_size[1] + 1)
            w1 = np.random.randint(0, padded_shape[2] - self.output_size[2] + 1)
        else:
            # 根据选中的不确定性中心点计算裁剪起始坐标
            center_d, center_h, center_w = center_coords
            d1 = center_d - self.output_size[0] // 2
            h1 = center_h - self.output_size[1] // 2
            w1 = center_w - self.output_size[2] // 2

            # 边界检查
            d1 = np.clip(d1, 0, padded_shape[0] - self.output_size[0])
            h1 = np.clip(h1, 0, padded_shape[1] - self.output_size[1])
            w1 = np.clip(w1, 0, padded_shape[2] - self.output_size[2])

        # --- 步骤 3: 应用裁剪 ---
        ret_dict = {}
        for key, item in padded_sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                ret_dict[key] = item[d1:d1 + self.output_size[0], 
                                     h1:h1 + self.output_size[1], 
                                     w1:w1 + self.output_size[2]]
            else:
                ret_dict[key] = item # 保留非3D数据

        return ret_dict
