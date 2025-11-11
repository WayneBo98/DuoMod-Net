import torch
import numpy as np
import torch.nn.functional as F
import random


class CenterCrop(object):
    '''
    Crops the center of the image in a sample.
    Handles cases where the input image is smaller than the output size by padding first.
    
    Args:
        output_size (tuple or int): Desired output size. If int, a cubic crop is made.
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = tuple(output_size)

    def __call__(self, sample):
        image = sample['image']
        original_shape = image.shape
        
        # --- 步骤 1: 对所有数据进行必要的填充 ---
        padded_sample = {}
        padding_values = []
        
        # 计算每个维度需要填充的总量
        for i in range(3):
            # 如果原始尺寸小于目标尺寸，计算需要填充的总宽度
            pad_total = max(0, self.output_size[i] - original_shape[i])
            # 将填充量均分到前后
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            padding_values.append((pad_before, pad_after))

        # 对sample中的每一个item应用相同的padding
        for key, item in sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                padded_sample[key] = np.pad(item, padding_values, mode='constant', constant_values=0)
            else:
                padded_sample[key] = item
                
        padded_image = padded_sample['image']
        padded_shape = padded_image.shape

        # --- 步骤 2: 基于填充后的图像计算中心坐标 ---
        # 此时 padded_shape 肯定大于或等于 output_size，计算结果永远是正数
        d1 = (padded_shape[0] - self.output_size[0]) // 2
        h1 = (padded_shape[1] - self.output_size[1]) // 2
        w1 = (padded_shape[2] - self.output_size[2]) // 2
        
        # --- 步骤 3: 对所有填充后的数据应用中心裁剪 ---
        ret_dict = {}
        for key, item in padded_sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                ret_dict[key] = item[d1:d1 + self.output_size[0], 
                                     h1:h1 + self.output_size[1], 
                                     w1:w1 + self.output_size[2]]
            else:
                ret_dict[key] = item
                
        return ret_dict

class RandomCrop(object):
    '''
    Crop randomly the image in a sample, with a high probability of sampling from the foreground.
    Handles cases where the input image is smaller than the desired output size by padding first.
    
    Args:
        output_size (tuple or int): Desired output size. If int, a cubic crop is made.
        fg_ratio (float): Probability of sampling a patch centered on a foreground voxel.
    '''
    def __init__(self, output_size, fg_ratio=0.8): # 新增 fg_ratio 参数
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = tuple(output_size)
        
        assert isinstance(fg_ratio, float) and 0 <= fg_ratio <= 1
        self.fg_ratio = fg_ratio # 保存前景采样率

    def __call__(self, sample):
        # sample 字典必须同时包含 'image' 和 'label'
        image, label = sample['image'], sample['label']
        original_shape = image.shape
        
        # --- 步骤 1: 对所有需要处理的数据进行填充 (这部分逻辑来自你的代码，是正确的，予以保留) ---
        padded_sample = {}
        padding_values = []
        
        for i in range(3):
            if original_shape[i] < self.output_size[i]:
                pad_width = (self.output_size[i] - original_shape[i]) // 2 + 3
                padding_values.append((pad_width, pad_width))
            else:
                padding_values.append((0, 0))
        
        for key, item in sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                # 使用 reflect 模式填充可以减少边界伪影，constant 也可以
                padded_sample[key] = np.pad(item, padding_values, mode='constant', constant_values=0) 
            else:
                padded_sample[key] = item
        
        padded_image = padded_sample['image']
        padded_label = padded_sample['label'] # 获取填充后的label
        padded_shape = padded_image.shape
        
        # --- 步骤 2: 基于填充后的图像，根据策略计算裁剪坐标 ---
        
        # 默认执行随机采样 (作为找不到前景时的备用方案)
        perform_fg_sampling = True
        
        if random.random() < self.fg_ratio:
            # 尝试进行前景采样
            fg_indices = np.argwhere(padded_label > 0)
            
            if len(fg_indices) > 0:
                # 随机选择一个前景点作为裁剪中心
                center_d, center_h, center_w = random.choice(fg_indices)

                # 从中心点计算裁剪的起始点
                d1 = center_d - self.output_size[0] // 2
                h1 = center_h - self.output_size[1] // 2
                w1 = center_w - self.output_size[2] // 2

                # --- 边界检查，确保裁剪框不会超出图像范围 ---
                d1 = max(0, d1)
                h1 = max(0, h1)
                w1 = max(0, w1)

                # 确保加上patch size后不会越界
                d1 = min(d1, padded_shape[0] - self.output_size[0])
                h1 = min(h1, padded_shape[1] - self.output_size[1])
                w1 = min(w1, padded_shape[2] - self.output_size[2])
                
                perform_fg_sampling = False # 成功执行了前景采样
            # 如果没有前景点，则会自动执行下面的随机采样
        
        if perform_fg_sampling: # 如果不进行前景采样 或 前景采样失败
             # 纯随机采样 (这部分逻辑来自你的代码)
            d1 = np.random.randint(0, padded_shape[0] - self.output_size[0] + 1)
            h1 = np.random.randint(0, padded_shape[1] - self.output_size[1] + 1)
            w1 = np.random.randint(0, padded_shape[2] - self.output_size[2] + 1)

        # --- 步骤 3: 对所有填充后的数据应用相同的裁剪 (这部分逻辑来自你的代码) ---
        ret_dict = {}
        for key, item in padded_sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                ret_dict[key] = item[d1:d1 + self.output_size[0], 
                                     h1:h1 + self.output_size[1], 
                                     w1:w1 + self.output_size[2]]
            else:
                ret_dict[key] = item

        return ret_dict

class ClassBalancedCrop(object):
    """
    在随机裁剪时，优先采样稀有类别。

    Args:
        output_size (tuple or int): 期望的输出尺寸。
        num_classes (int): 类别总数。
        class_probabilities (np.ndarray): 由预计算函数得到的每个类别的采样概率。
        fg_ratio (float): 执行前景采样的总概率。
    """
    def __init__(self, output_size, num_classes, class_probabilities, fg_ratio=0.8):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = tuple(output_size)
            
        self.num_classes = num_classes
        # 确保传入的概率数组长度正确
        assert len(class_probabilities) == num_classes
        self.class_probabilities = class_probabilities
        
        assert isinstance(fg_ratio, float) and 0 <= fg_ratio <= 1
        self.fg_ratio = fg_ratio

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # --- 步骤 1: 填充图像 (逻辑与您原版相同) ---
        padded_sample = {}
        padding_values = []
        original_shape = image.shape
        
        for i in range(3):
            pad_needed = self.output_size[i] - original_shape[i]
            if pad_needed > 0:
                pad_width = pad_needed // 2 + 3 # 增加一些余量
                padding_values.append((pad_width, pad_width))
            else:
                padding_values.append((0, 0))
        
        for key, item in sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                padded_sample[key] = np.pad(item, padding_values, mode='constant', constant_values=0)
            else:
                padded_sample[key] = item
                
        padded_label = padded_sample['label']
        padded_shape = padded_label.shape
        
        # --- 步骤 2: 类别均衡的中心点选择 (核心改动) ---
        center_d, center_h, center_w = None, None, None

        if random.random() < self.fg_ratio:
            # 1. 找出当前图像中存在哪些前景类别
            present_classes = np.unique(padded_label)
            present_fg_classes = present_classes[present_classes > 0]

            if len(present_fg_classes) > 0:
                # 2. 从存在的类别中，根据预计算的全局采样概率，选择一个目标类别
                # 我们只考虑当前图像中存在的类别
                p = self.class_probabilities[present_fg_classes]
                p_normalized = p / p.sum() # 归一化，使当前可选类别的概率和为1
                
                target_class = np.random.choice(present_fg_classes, p=p_normalized)
                
                # 3. 找出目标类别的所有体素坐标
                class_indices = np.argwhere(padded_label == target_class)
                
                if len(class_indices) > 0:
                    # 4. 从中随机选择一个作为中心点
                    center_d, center_h, center_w = random.choice(class_indices)

        # 如果前景采样失败 (例如图像中无前景，或随机数未命中fg_ratio)
        if center_d is None:
            # 执行完全随机采样
            d1 = np.random.randint(0, padded_shape[0] - self.output_size[0] + 1)
            h1 = np.random.randint(0, padded_shape[1] - self.output_size[1] + 1)
            w1 = np.random.randint(0, padded_shape[2] - self.output_size[2] + 1)
        else:
            # 根据选中的中心点计算裁剪起始坐标
            d1 = center_d - self.output_size[0] // 2
            h1 = center_h - self.output_size[1] // 2
            w1 = center_w - self.output_size[2] // 2

            # 边界检查
            d1 = np.clip(d1, 0, padded_shape[0] - self.output_size[0])
            h1 = np.clip(h1, 0, padded_shape[1] - self.output_size[1])
            w1 = np.clip(w1, 0, padded_shape[2] - self.output_size[2])

        # --- 步骤 3: 应用裁剪 (逻辑与您原版相同) ---
        ret_dict = {}
        for key, item in padded_sample.items():
            if isinstance(item, np.ndarray) and item.ndim == 3:
                ret_dict[key] = item[d1:d1 + self.output_size[0], 
                                     h1:h1 + self.output_size[1], 
                                     w1:w1 + self.output_size[2]]
            else:
                ret_dict[key] = item
                
        return ret_dict

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, do_flip):
        # 将随机决定（do_flip）作为参数传入，保证对一个sample的所有项翻转与否是一致的
        if do_flip:
            img = np.flip(img, 2).copy() # LR flip is along axis 1 (width)
        return img

    def __call__(self, sample):
        # 在处理整个sample前，先做一次随机决定
        do_flip = random.random() <= self.prob
        
        processed_sample = {}
        for key, item in sample.items():
            # !!!!!!!!!!!!!!!!!!! 关键修正 !!!!!!!!!!!!!!!!!!!
            # 在执行翻转操作前，检查item是否为3D numpy数组
            if isinstance(item, np.ndarray) and item.ndim == 3:
                # 如果是，则执行翻转
                processed_sample[key] = self._flip(item, do_flip)
            else:
                # 如果不是（例如是case_id字符串），则原样保留
                processed_sample[key] = item
                
        return processed_sample

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, do_flip):
        if do_flip:
            img = np.flip(img, 1).copy() # 假设 axis 2 是您定义的上下翻转轴
        return img

    def __call__(self, sample):
        do_flip = random.random() <= self.prob
        
        processed_sample = {}
        for key, item in sample.items():
            # !!!!!!!!!!!!!!!!!!! 关键修正 !!!!!!!!!!!!!!!!!!!
            # 同样加入类型检查
            if isinstance(item, np.ndarray) and item.ndim == 3:
                processed_sample[key] = self._flip(item, do_flip)
            else:
                processed_sample[key] = item
                
        return processed_sample


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        ret_dict = {}
        for key, item in sample.items():
            if key in ['image', 'label']:
                if key == 'image':
                    # 将图像转换为 (1, D, H, W) 的 float tensor
                    ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
                elif key == 'label':
                    # 将标签转换为 (D, H, W) 的 long tensor
                    ret_dict[key] = torch.from_numpy(item).long()
            # !!!!!!!!!!!!!!!!!!! 关键修正 !!!!!!!!!!!!!!!!!!!
            # 对于不认识的key（如case_id），直接保留其原始值，而不是报错
            else:
                ret_dict[key] = item

        return ret_dict
